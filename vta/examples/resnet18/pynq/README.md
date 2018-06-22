# Resnet-18 Example on Pynq-based VTA Design

Follow the first two parts of the [Installation Guide](../../../docs/how_to/install.md) to make sure that the VTA python libraries are installed, and that the RPC server is running on the Pynq FPGA dev board.

We recommend leaving the `config.json` to its default parameterization (of course you can change the target between "sim" and "pynq").

Simply run the example program. We rely on pickle to store parameters which now only works with python2.
```bash
python2 imagenet_predict.py
```

The script will first download the following files into `_data/` directory:
* `cat.jpg` which provides a test sample for the ImageNet classifier
* `quantize_graph.json` which describes the NNVM graph of the 8-bit ResNet-18
* `quantize_params.plk` which contains the network parameters
* `synset.txt` which contains the ImageNet categories

Next, it will run imagenet classification using the ResNet18 architecture on a VTA design that performs 8-bit integer inference, to perform classification on a cat image `cat.jpg`.

The script reports runtime measured on the Pynq board (in seconds), and the top-1 result category:
```
('x', (1, 3, 224, 224))
Build complete...
('TVM prediction top-1:', 281, 'tabby, tabby cat')
t-cost=0.41906
```
