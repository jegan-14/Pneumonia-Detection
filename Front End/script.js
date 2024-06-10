$(document).ready(function () {
    classify();
});

function classify(){
    const btn = document.getElementById("btn");
    const fileInput = document.getElementById("fileInput");
    
    fileInput.addEventListener("change", function() {
        updateText();
    });

    btn.addEventListener("click", ()=>{
        if(fileInput.files.length <= 0){
            alert("Select any image");
        }
        else{
            const img = fileInput.files[0];
            const reader = new FileReader();
            reader.addEventListener("load",()=>{
                var data = reader.result;
                var api = "http://127.0.0.1:5000/classify_image";
                $.post(
                    api,{
                        "image_data" : data
                    },
                    function(data,status){
                        const predictedClass = data["class"];
                        var predictionProb = data["pred_prob"];
                        if(data!=null){
                            var result = document.getElementById("result");
                            if(predictionProb < 0.5){
                                predictionProb = 1 - predictionProb;
                            }
                            predictionProb = Math.round(predictionProb*100);
                            result.innerText = `The predicted image is ${predictedClass} with the probability of ${predictionProb}%`;
                        }
                    }
                )
            })
            reader.readAsDataURL(img);
        }
    });
}

function updateText() {
    let fileInput = document.getElementById('fileInput');
    let fileName = "";
    if (fileInput.files.length > 0) {
        fileName = fileInput.files[0].name;
    } 
    document.getElementById('selectedImageText').innerText = fileName + " is selected.";
}
