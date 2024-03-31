document.addEventListener("DOMContentLoaded", function() {
    var xhr = new XMLHttpRequest();
    xhr.open("GET", "list_images.php", true);
    xhr.onreadystatechange = function() {
        if (xhr.readyState == 4 && xhr.status == 200) {
            var images = JSON.parse(xhr.responseText);
            displayImages(images);
        }
    };
    xhr.send();
});

function displayImages(images) {
    var imageContainer = document.getElementById("imageContainer");
    images.forEach(function(image) {
        var imgElement = document.createElement("img");
        imgElement.src = image;
        imgElement.alt = "Image";
        imgElement.width = 200; // Adjust width as needed
        imgElement.height = 200; // Adjust height as needed
        imageContainer.appendChild(imgElement);
    });
}
