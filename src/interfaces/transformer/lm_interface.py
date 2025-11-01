import torch
import torch.nn
from typing import Dict, Tuple
from interfaces.model_interface import ModelInterface
from interfaces.encoder_decoder import EncoderDecoderResult
from models.transformer_enc_dec import TransformerResult
from models.encoder_decoder import add_eos, add_eos_pack
import layers
import pdb


class TransformerLMInterface(ModelInterface):
    def __init__(self, model: torch.nn.Module, label_smoothing: float = 0.0):
        self.model = model
        self.label_smoothing = label_smoothing
        self.encoder_sos = self.model.encoder_sos
        self.encoder_eos = self.model.encoder_eos

    def loss(
        self,
        outputs: TransformerResult,
        ref: torch.Tensor,
        mask: torch.Tensor,
        normalize,
    ) -> torch.Tensor:
        l = layers.cross_entropy( 
            # TODO: Set the ignore_index in a cleaner way 
            # (e.g. linked to Vocabulary settings)
            outputs.data, ref, reduction="none", smoothing=self.label_smoothing, ignore_index=0
        )
        l = l.reshape_as(ref) * mask
        if normalize:
            return l.sum() / mask.sum()
        else:
            return l.sum()

    def decode_outputs(
        self, outputs: EncoderDecoderResult
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return outputs.outputs, outputs.out_lengths

    # todo, this will return hidden states as well, as required
    def __call__(
        self, data: Dict[str, torch.Tensor], normalize=True, pack=False, nli=False,
        regularize=False, layer_id=-1, sci_heads=-1
    ) -> EncoderDecoderResult:
        in_len = data["in_len"].long() + (1 - int(pack)) # in_len for packing included SOS

        # will skip this with packing
        if not pack:
            sos_tensor = torch.ones((1, data["in"].shape[1])) * self.encoder_sos
            sos_tensor = sos_tensor.to(data["in"].device)
            inp_data = torch.cat(
                [sos_tensor, data["in"]],
                dim=0,
            ).transpose(0, 1)
        else:
            # Handled by the PackingDataset
            inp_data = data["in"].transpose(0,1)
        # print(inp_data)

        labels_key = "in"

        if not pack:
            out_data = add_eos(
                data[labels_key], data["in_len"], self.encoder_eos
            ).transpose(0, 1)
        else:
            out_data = add_eos_pack(
                data[labels_key], data["in_len"], self.encoder_eos, self.encoder_sos
            ).transpose(0, 1)
        # print(out_data)

        # recreate targets with packing

        # inp_data =  bs x seq_len: [SOS] a b c
        # out_data =  bs x seq_len e.g.  a b c [EOS]
        res = self.model(inp_data, in_len, get_hidden_states = regularize, layer_id = layer_id, sci_heads = sci_heads)
        # pdb.set_trace()

        res.data = res.data.transpose(0, 1)
        real_model = self.model.module if hasattr(self.model, "module") else self.model
        len_mask = ~real_model.generate_len_mask(inp_data.shape[1], in_len).transpose(0, 1)

        loss = self.loss(res, out_data.transpose(0, 1), len_mask, normalize)
        return EncoderDecoderResult(res.data, res.length, loss, res.hidden_states)

class TransformerHFInterface(ModelInterface):
    def __init__(self, model: torch.nn.Module, label_smoothing: float = 0.0, encoder_sos=-1, encoder_eos=-1):
        self.model = model
        self.label_smoothing = label_smoothing
        self.encoder_sos = encoder_sos
        self.encoder_eos = encoder_eos

    def loss(
        self,
        outputs: TransformerResult,
        ref: torch.Tensor,
        mask: torch.Tensor,
        normalize,
    ) -> torch.Tensor:
        l = layers.cross_entropy( 
            # TODO: Set the ignore_index in a cleaner way 
            # (e.g. linked to Vocabulary settings)
            outputs, ref, reduction="none", smoothing=self.label_smoothing, ignore_index=0
        )
        l = l.reshape_as(ref) * mask
        if normalize:
            return l.sum() / mask.sum()
        else:
            return l.sum()

    def generate_mask(self, max_len, in_lens):
        return torch.arange(max_len).expand(len(in_lens), max_len).to(in_lens.device) < in_lens.unsqueeze(1)

    def decode_outputs(
        self, outputs: EncoderDecoderResult
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return outputs.outputs, outputs.out_lengths

    def __call__(
        self, data: Dict[str, torch.Tensor], normalize=True, pack=False, nli=False,
        regularize=False, layer_id=-1, sci_heads=-1
    ) -> EncoderDecoderResult:
        in_len = data["in_len"].long() + (1 - int(pack)) # in_len for packing included SOS

        # will skip this with packing
        if not pack:
            sos_tensor = torch.ones((1, data["in"].shape[1]), dtype=torch.long) * self.encoder_sos
            sos_tensor = sos_tensor.to(data["in"].device)
            inp_data = torch.cat(
                [sos_tensor, data["in"]],
                dim=0,
            ).transpose(0, 1)
        else:
            inp_data = data["in"].transpose(0,1)
        # print(inp_data)

        labels_key = "in"

        if not pack:
            out_data = add_eos(
                data[labels_key], data["in_len"], self.encoder_eos
            ).transpose(0, 1)
        else:
            out_data = add_eos_pack(
                data[labels_key], data["in_len"], self.encoder_eos, self.encoder_sos
            ).transpose(0, 1)
        # print(out_data)

        # recreate targets with packing

        # inp_data =  bs x seq_len: [SOS] a b c
        # out_data =  bs x seq_len e.g.  a b c [EOS]
        attn_mask = self.generate_mask(inp_data.shape[1], in_len).to(inp_data.device)
        out = self.model(inp_data, attention_mask=attn_mask, output_hidden_states=regularize)

        res = out["logits"]
        res = res.transpose(0, 1)
        len_mask = attn_mask.transpose(
            0, 1
        )
        if regularize:
            proportion = int(sci_heads * out.hidden_states[0].shape[-1])
            hidden_states = out.hidden_states[layer_id][:, :, :proportion]
        else:
            hidden_states = None

        if nli:
            tgt_list = torch.tensor([29907,29909,29933]).to(inp_data
.device)
            tgts = data["in"][data["in_len"].int()-1,torch.arange(data["in_len"].shape[0])]
            curr = res[data["in_len"].int()-1, torch.arange(data["in_len"].shape[0]), :]
            curr = curr[:, tgt_list]
            gold_tgts = torch.searchsorted(tgt_list, tgts)
            loss = layers.cross_entropy(curr, gold_tgts, reduction="none", smoothing=self.label_smoothing, ignore_index=0)
            ret_loss = loss.sum()
            return EncoderDecoderResult(res, in_len, ret_loss, hidden_states)
        else:
            loss = self.loss(res, out_data.transpose(0, 1), len_mask, normalize)
            return EncoderDecoderResult(res, in_len, loss, hidden_states)