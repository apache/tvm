# Resnet-18 Example on Pynq-based VTA Design

Follow the first two parts of the [Installation Guide](../../../docs/how_to/install.md) to make sure that the VTA python libraries are installed, and that the RPC server is running on the Pynq FPGA dev board.

Simply run the following python script:
```bash
python imagenet_predict.py
```

This will run imagenet classification using the ResNet18 architecture on a VTA design that performs 8-bit integer inference, to perform classification on a cat image `cat.jpg`.

The script reports runtime measured on the Pynq board (in seconds), and the top-1 result category:
```
('x', (1, 3, 224, 224))
Build complete...
('TVM prediction top-1:', 281, 'tabby, tabby cat')
t-cost=0.41906
```
