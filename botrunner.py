import clip, os, numpy as np, torch.nn.functional as nnf, sys, skimage.io as io, PIL.Image, torch, requests, shutil, hikari, lightbulb, glob
from torch import nn
from typing import Tuple, List, Union, Optional
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AdamW,
    get_linear_schedule_with_warmup
)
N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]
D = torch.device
CPU = torch.device('cpu')
cwd = os.getcwd()
savepath = "pretrained_models"
os.makedirs(savepath, exist_ok=True)
modelpath = os.path.join(savepath, 'coco_weights.pt')
class MLP(nn.Module):
    def forward(self, x: T) -> T:
        return self.model(x)
    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) -1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)
class ClipCaptionModel(nn.Module):
    def get_dummy_token(self, batch_size: int, device: D) -> T:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)
    def forward(self, tokens: T, prefix: T, mask: Optional[T] = None, labels: Optional[T] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out
    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 10:
            self.clip_project = nn.Linear(prefix_size, self.gpt_embedding_size * prefix_length)
        else:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2, self.gpt_embedding_size * prefix_length))
class ClipCaptionPrefix(ClipCaptionModel):
    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()
    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self

def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None, entry_length=67, temperature=1., stop_token: str = '.'):
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


device = 'cpu'
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

prefixlength = 10
model = ClipCaptionModel(prefixlength)
model.load_state_dict(torch.load(modelpath, map_location=CPU))
model = model.eval()
device = 'cpu'
model = model.to(device)

bot = lightbulb.BotApp("") #<- replace with your own token here.
@bot.listen(hikari.StartedEvent)
async def on_ready(event):
    print("Ready!")

@bot.command
@lightbulb.option("showimage", "Shows the image you chose upon finishing. Default is False.", type=bool, default=False)
@lightbulb.option("showtopresult", "Shows only the top result. Default is True.", type=bool, default=True)
@lightbulb.option("temp", "Self explanatory. Max is 4", type=int)
@lightbulb.option("link", "The image link you want to use.", type=str)
@lightbulb.command("identify", "Uses an image to text AI to detect what is in the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def imageDetection(ctx):
    if ctx.options.showimage == True:
        imageurl = ctx.options.link
        filename = imageurl.split("/")[-1]
        r = requests.get(imageurl, stream=True)
        if r.status_code == 200:
            r.raw.decode_content = True
            with open(filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        image = io.imread(filename)
        pil = PIL.Image.fromarray(image)
        await ctx.respond("Here is your image!")
        imgfile = hikari.File(filename)
        await ctx.respond(imgfile)
    if ctx.options.temp > 5:
        await ctx.respond("Temp is too high! Please pick a lower number.")
        return
    await ctx.respond("Detecting...")
    imageurl = ctx.options.link
    filename = imageurl.split("/")[-1]
    r = requests.get(imageurl, stream=True)
    if r.status_code == 200:
        r.raw.decode_content = True
        with open(filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    image = io.imread(filename)
    pil = PIL.Image.fromarray(image)
    image = preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        prefixembed = model.clip_project(prefix).reshape(1, prefixlength, -1)
        if ctx.options.showtopresult == False:
            generatedtextembed = generate_beam(model, tokenizer, embed=prefixembed)
            await ctx.respond(f"showTopResult set to False. Top 5 results for your image were:\n {generatedtextembed}")
            return
        else:
            generatedtextembed = generate_beam(model, tokenizer, embed=prefixembed)[0]
            await ctx.respond(f"showTopResult set to True. The top result is: {generatedtextembed}")

for i in glob.glob("*.jpg"):
    if os.path.exists(i):
        os.remove(i)
for q in glob.glob("*.png"):
    if os.path.exists(q):
        os.remove(q)

@bot.command
@lightbulb.option("link", "The link to test for.", type=str)
@lightbulb.command("linktest", "Test a link before using the identify command.")
@lightbulb.implements(lightbulb.SlashCommand)
async def linktester(ctx):
    link = ctx.options.link
    if link.endswith(".jpg"):
        await ctx.respond("This link is a jpg image. This will work!")
    elif link.endswith(".png"):
        await ctx.respond("This link is a png image. This will work!")
    elif link.endswith(".webp"):
        await ctx.respond("This link is a webp image. This will **not** work, however, a fix for this is planned.")
    elif link.endswith(".gif"):
        await ctx.respond("This link is a gif image. This will **not** work, and a fix is not planned.")
bot.run()
