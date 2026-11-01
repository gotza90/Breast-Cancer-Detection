function uploadImage(){

let fileInput = document.getElementById("imageUpload")

if(fileInput.files.length === 0){

alert("Please upload image")

return

}

let file = fileInput.files[0]

let formData = new FormData()

formData.append("image",file)

document.getElementById("prediction").innerHTML="Analyzing..."

fetch("/predict",{

method:"POST",
body:formData

})

.then(res=>res.json())

.then(data=>{

console.log(data)

// prediction
document.getElementById("prediction").innerHTML=data.prediction


// confidence
let prob = Number(data.probability)

let percent = prob * 100

document.getElementById("confidence").innerHTML =
"Confidence: "+percent.toFixed(2)+"%"


// progress bar
document.getElementById("confidence-fill").style.width =
percent+"%"


// original image
document.getElementById("original").src =
URL.createObjectURL(file)


// gradcam
document.getElementById("gradcam").src =
data.heatmap


// overlay
document.getElementById("overlay").src =
data.overlay

})

}