# Image-Identifier-Discord-Bot
A discord bot that uses GPT2 to identify an image from a link.
## How to use

In line 139 of botrunner.py, insert your own bots token.
```python
bot = lightbulb.BotApp("") #<- replace with your own token here.
```

## Requirements
You will need some requirements to get started:

- The COCO weights [dataset](https://drive.google.com/file/d/1ht1sOOM5h6vYnhzytwTCxqUjyR8DbAOT/view?usp=sharing)

- Python 3.8.6 with PyTorch + CUDA installed. PyTorch CPU may work too if you don't have enough GPU power.

- Install the packages from requirements.txt.

### Feel free to improve or make this better, I'm just dumping this here for the sake of clearing some space up.
