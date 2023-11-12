
function dragOver(event) {
    event.preventDefault();
}

function drop(event) {
    event.preventDefault();
    var files = event.dataTransfer.files;
    var uploadFile = document.getElementById("uploadFile");

    if (files.length > 0) {
        uploadFile.files = files;
    }
}
