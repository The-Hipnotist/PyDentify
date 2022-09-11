# Image Identifier Discord Bot (AKA PyDentify)
A discord bot that uses GPT2 to identify an image from a link.
## How to use

In line 139 of botrunner.py, insert your own bots token.
```python
bot = lightbulb.BotApp("") #<- replace with your own token here.
```

## Requirements
You will need some requirements to get started:

- The COCO weights [dataset](http://images.cocodataset.org/zips/test2014.zip)

- Python 3.8.6 with PyTorch + CUDA 11.3 installed. PyTorch CPU may work too if you don't have enough GPU power.

- Install the packages from requirements.txt.

### Feel free to improve or make this better, I'm just dumping this here for the sake of clearing some space up.
