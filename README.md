# Image Identifier Discord Bot (AKA PyDentify)
A discord bot that uses GPT2 to identify an image from a link.
## How to use

In line 139 of botrunner.py, insert your own bots token.
```python
bot = lightbulb.BotApp("") #<- replace with your own token here.
```
Then, from line 145 to 150, feel free to add your own commands.
```python
@lightbulb.option("showimage", "Shows the image you chose upon finishing. Default is False.", type=bool, default=False)
@lightbulb.option("showtopresult", "Shows only the top result. Default is True.", type=bool, default=True)
@lightbulb.option("temp", "Self explanatory. Max is 4", type=int)
@lightbulb.option("link", "The image link you want to use.", type=str)
@lightbulb.command("identify", "Uses an image to text AI to detect what is in the image.")
@lightbulb.implements(lightbulb.SlashCommand)
```
**Do not** adjust lines 151 to 195, those are necessary for the bot and AI to run.
```python
async def imageDetection(ctx): 
    if ctx.options.showimage == True: #<- Only adjust this if you are not using a bool type, or if this is set to False.
        ... #<- DO NOT ADJUST THESE LINES ONWARD!
```
However you **can** adjust lines 197 to 210 if you want.
```python
@lightbulb.option("link", "The link to test for.", type=str)
@lightbulb.command("linktest", "Test a link before using the identify command.")
@lightbulb.implements(lightbulb.SlashCommand)
async def linktester(ctx): #<- This is ok though.
    link = ctx.options.link
    if link.endswith(".jpg"):
        await ctx.respond("This link is a jpg image. This will work!")
    elif link.endswith(".png"):
        await ctx.respond("This link is a png image. This will work!")
    elif link.endswith(".webp"):
        await ctx.respond("This link is a webp image. This will **not** work, however, a fix for this is planned.")
    elif link.endswith(".gif"):
        await ctx.respond("This link is a gif image. This will **not** work, and a fix is not planned.")
```
## Requirements
You will need some requirements to get started:

~~- The COCO weights [dataset](https://drive.google.com/file/d/1ht1sOOM5h6vYnhzytwTCxqUjyR8DbAOT/view?usp=sharing)~~

~~Google Drive link has been taken down, an alternative might not be found.~~

New link was actually found, this is the new link: https://mega.nz/file/q6ISxJbS#BGEh5R6Wy8pXVWfNwKpiN9Wr1OGk8iXVrDMkaYoo01I

- Python 3.8.6 with PyTorch + CUDA 11.3 installed. PyTorch CPU may work too if you don't have enough GPU power.
- To test if you have GPU installed with Pytorch, run:
```python
import torch
print(torch.cuda.is_available())
```
or:
```python
from torch.cuda import is_available
print(is_available())
```
If you get an output that just says True, you have cuda installed. If you get False, GPU is not installed.
- Install the packages from requirements.txt.

### Feel free to improve or make this better, I'm just dumping this here for the sake of clearing some space up.
