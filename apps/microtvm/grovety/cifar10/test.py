import tensorflow as tf
import numpy as np



def get_data():
    import pickle
    import os, pathlib

    current_dir = pathlib.Path(os.path.dirname(__file__)).resolve()
    with open(current_dir / 'data/batches.meta', 'rb') as file:
        label_names = pickle.load(file, encoding='bytes')

    with open(current_dir / 'data/test_batch', 'rb') as file:
        test_batch = pickle.load(file, encoding='bytes')

    dataset = []
    i = 0
    while len(dataset) < 20:
        label = test_batch[b'labels'][i]
        label_str = label_names[b'label_names'][label].decode('UTF-8')
        data = test_batch[b'data'][i]
        i += 1
        if label_str not in ['cat', 'dog', 'frog']:
            continue
        dataset.append((label_str, data))

    return dataset

def grayConversion(image):
    grayValue = 0.07 * image[:,:,2] + 0.72 * image[:,:,1] + 0.21 * image[:,:,0]
    gray_img = grayValue.astype(np.uint8)
    return gray_img

if __name__ == '__main__':
    import PIL

    interpreter = tf.lite.Interpreter(model_path='cifar_quant_8bit.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]


    pred_labels = ['cat', 'dog', 'frog']

    for label, data in get_data():
        img = data.reshape(3, 32, 32)
        img = img.transpose(1, 2, 0)

        img = grayConversion(img)

        # from matplotlib import pyplot as plt
        # plt.imshow(grayConversion(img))
        # plt.show()

            # input_scale, input_zero_point = input_details["quantization"]
            # test_image = test_image / input_scale + input_zero_point
        # data = data.reshape((1, 32, 32))
        img = np.expand_dims(img, axis=0).astype(input_details["dtype"])
        img = np.expand_dims(img, axis=3).astype(input_details["dtype"])

        interpreter.set_tensor(input_details["index"], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]
        print(f"{label} =====> {pred_labels[output.argmax()]}   {output}")
