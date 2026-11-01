from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# load model
model = tf.keras.models.load_model("model/breast_cancer_model.keras")

IMG_SIZE = 50
THRESHOLD = 0.5


# force build model
model.predict(np.zeros((1,IMG_SIZE,IMG_SIZE,3)))


# print layers (for debugging)
for layer in model.layers:
    print(layer.name, layer.__class__.__name__)


# find last conv layer automatically
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

    last_conv_layer = model.get_layer(last_conv_layer_name)

    # model لحد conv layer
    conv_model = tf.keras.Model(
        model.inputs,
        last_conv_layer.output
    )

    # باقي الموديل بعد conv
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])

    x = classifier_input

    for layer in model.layers[model.layers.index(last_conv_layer)+1:]:
        x = layer(x)

    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:

        conv_outputs = conv_model(img_array)

        tape.watch(conv_outputs)

        preds = classifier_model(conv_outputs)

        loss = preds[0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs * pooled_grads

    heatmap = tf.reduce_sum(heatmap, axis=-1)

    heatmap = tf.maximum(heatmap,0)

    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    file_bytes = np.frombuffer(file.read(), np.uint8)

    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    original = img.copy()

    # preprocess
    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))

    img = img / 255.0

    img = np.expand_dims(img,axis=0)

    # prediction
    pred = model.predict(img)[0][0]

    label = "Malignant" if pred >= THRESHOLD else "Benign"

    print("Prediction:", pred)


    # find last conv layer
    last_conv = "conv2d_2"

    print("Using conv layer:", last_conv)

   

    # GradCAM
    heatmap = make_gradcam_heatmap(img, model, last_conv)

    print("Heatmap max:", np.max(heatmap))


   # resize heatmap
    heatmap = cv2.resize(heatmap,(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_CUBIC)

    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = cv2.resize(original,(IMG_SIZE,IMG_SIZE))

    overlay = cv2.addWeighted(original,0.6,heatmap_color,0.4,0)

    print("Heatmap shape:", heatmap.shape)
    print("Heatmap max:", np.max(heatmap))

    # save images
    heatmap_path = os.path.join(app.root_path,"static","gradcam_heatmap.jpg")

    overlay_path = os.path.join(app.root_path,"static","gradcam_overlay.jpg")


    cv2.imwrite(heatmap_path, heatmap_color)

    cv2.imwrite(overlay_path, overlay)

    print("Heatmap min:", np.min(heatmap))
    print("Heatmap max:", np.max(heatmap))

    print("GradCAM saved")


    return jsonify({

        "prediction": label,

        "probability": float(pred),

        "heatmap": "/static/gradcam_heatmap.jpg?" + str(np.random.rand()),

        "overlay": "/static/gradcam_overlay.jpg?" + str(np.random.rand())

    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)