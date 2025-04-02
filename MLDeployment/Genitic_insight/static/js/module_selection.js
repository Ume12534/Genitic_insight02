document.getElementById('dataset-upload').addEventListener('change', function(event) {
    var file = event.target.files[0];
    if (file && file.type === "text/csv") {
        document.getElementById('algorithm-list').style.display = 'block';
        document.getElementById('train-button').disabled = false;
    } else {
        alert("Please upload a valid CSV file.");
    }
});

document.getElementById('train-button').addEventListener('click', function() {
    var selectedAlgorithm = document.querySelector('input[name="algorithm"]:checked').value;
    alert("Training model with " + selectedAlgorithm);
    // Here you would add the code to train the model with the selected algorithm
});