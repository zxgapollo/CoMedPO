"""LLaVA-Med adapter and answer-only likelihood evaluation."""

import torch
from torch import nn
from PIL import Image

from .loss import BRANCHES, comedpo_loss, sequence_logps


class LlavaCollator:
    def __init__(self, tokenizer, image_processor, conversation="mistral_instruct",
                 max_length=2048, image_start_end=False):
        self.tokenizer, self.processor = tokenizer, image_processor
        self.conversation, self.max_length = conversation, max_length
        self.image_start_end = image_start_end

    def _encode(self, question, answer):
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token
        marker = "<im_start><image><im_end>" if self.image_start_end else "<image>"
        conv = conv_templates[self.conversation].copy()
        conv.append_message(conv.roles[0], f"{marker}\n{question}")
        conv.append_message(conv.roles[1], None)
        prefix = tokenizer_image_token(conv.get_prompt(), self.tokenizer, return_tensors="pt")
        # Tokenize the response separately to make the prompt boundary explicit.
        answer_ids = self.tokenizer(answer.strip(), add_special_tokens=False).input_ids
        if not answer_ids:
            raise ValueError("The tokenizer produced an empty response")
        if answer_ids[-1] != self.tokenizer.eos_token_id:
            answer_ids.append(self.tokenizer.eos_token_id)
        ids = torch.cat([prefix, torch.tensor(answer_ids, dtype=torch.long)])
        if len(ids) > self.max_length:
            raise ValueError("Text exceeds max_length; shorten the example explicitly")
        labels = ids.clone()
        labels[:len(prefix)] = -100
        return ids, labels

    def _image(self, image):
        # Pad before CLIP resizing so its center crop cannot remove either panel.
        side = max(image.size)
        color = tuple(int(x * 255) for x in self.processor.image_mean)
        square = Image.new("RGB", (side, side), color)
        square.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        return self.processor.preprocess(square, return_tensors="pt")["pixel_values"][0]

    def __call__(self, rows):
        result = {}
        for column, name in enumerate(BRANCHES):
            encoded = [self._encode(*row[column][:2]) for row in rows]
            ids, labels = zip(*encoded)
            lengths = torch.tensor([len(x) for x in ids])
            ids = nn.utils.rnn.pad_sequence(ids, batch_first=True,
                                            padding_value=self.tokenizer.pad_token_id)
            labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
            result[name] = {
                "input_ids": ids, "labels": labels,
                "attention_mask": torch.arange(ids.shape[1])[None, :] < lengths[:, None],
                "images": torch.stack([self._image(row[column][2]) for row in rows]),
            }
        return result


class CoMedPOPolicy(nn.Module):
    """Train new LoRA adapters against their frozen, adapter-disabled SFT base.

    Wrap the entire six-branch computation in one DDP forward. No adapter is
    merged or updated in the reference pass. Existing SFT adapters must be
    merged into the checkpoint before this wrapper is constructed.
    """

    def __init__(self, policy, beta=0.1, causal_weight=1.0, max_length=2048):
        super().__init__()
        self.policy = policy
        self.beta, self.causal_weight, self.max_length = beta, causal_weight, max_length

    def _score(self, batch):
        base = self.policy.get_base_model()
        images = batch["images"].to(dtype=base.get_vision_tower().dtype)
        # Use the SAME expanded labels as the forward inputs, including visual
        # patch positions. All base parameters are frozen by the LoRA setup.
        ids, positions, mask, past, embeds, labels = base.prepare_inputs_labels_for_multimodal(
            input_ids=batch["input_ids"], position_ids=None,
            attention_mask=batch["attention_mask"], past_key_values=None,
            labels=batch["labels"], images=images,
        )
        if labels.shape[1] > self.max_length:
            raise ValueError("Text plus image tokens exceeds max_length; no silent truncation is allowed")
        if embeds is not None and torch.is_grad_enabled():
            embeds.requires_grad_(True)
        output = self.policy(
            input_ids=ids, inputs_embeds=embeds, attention_mask=mask,
            position_ids=positions, use_cache=False, return_dict=True,
        )
        return sequence_logps(output.logits, labels)

    def forward(self, batch):
        was_training = self.policy.training
        self.policy.eval()
        try:
            with torch.no_grad(), self.policy.disable_adapter():
                reference = torch.stack([self._score(batch[name]) for name in BRANCHES], dim=1)
        finally:
            self.policy.train(was_training)
        policy = torch.stack([self._score(batch[name]) for name in BRANCHES], dim=1)
        return comedpo_loss(policy, reference, self.beta, self.causal_weight)
