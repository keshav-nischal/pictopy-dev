<?php
// Directory containing your images
$imageDirectory = '.\my-electron-app\images'

// Get all image files from the directory
$images = glob($imageDirectory . '*.{jpg,jpeg,png,gif}', GLOB_BRACE);

// Send images as JSON response
header('Content-Type: application/json');
echo json_encode($images);
?>
