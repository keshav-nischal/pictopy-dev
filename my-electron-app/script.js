


//..........................................images.................................................


  document.addEventListener('DOMContentLoaded', function() {
    const imageContainer = document.getElementById('image-container');
  
    // Function to shuffle the array randomly
    function shuffleArray(array) {
      for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
      }
      return array;
    }


    const images = [
        './images/images (1).jpeg',
        './images/images (2).jpeg',
        './images/images (3).jpeg',
        './images/images (4).jpeg',
        './images/images (5).jpeg',
        './images/images (6).jpeg',
        './images/images (7).jpeg',
        './images/images (8).jpeg',
        './images/images (9).jpeg',
        './images/images (10).jpeg',
        './images/images (11).jpeg',
        './images/images (12).jpeg',
        './images/images (13).jpeg',
        './images/images (14).jpeg',
        './images/images (15).jpeg',
        
        // Add more image URLs as needed
      ];
  
    // Function to create and display images randomly
    function displayRandomImages() {
      
  
      const shuffledImages = shuffleArray(images);
      // Clear the image container
      imageContainer.innerHTML = '';
      // Append shuffled images to the container
      shuffledImages.forEach(imageUrl => {
        const img = document.createElement('img');
        img.src = imageUrl;
        img.alt = 'Random Image';
        const div = document.createElement('div');
        div.classList.add('flex-item');
         // Make the last image larger
          
        
        div.appendChild(img);
        imageContainer.appendChild(div);
      });
    }
  
    // Call the function to display random images initially
    displayRandomImages();
  });
  
  //..........................images2......................
  document.addEventListener('DOMContentLoaded', function() {
    const imageContainer = document.getElementById('image-container1');
  
    // List of image URLs
    const imageUrls = [
        './images1/images (1).jpeg',
        './images1/images (2).jpeg',
        './images1/images (3).jpeg',
        './images1/images (4).jpeg',
        './images1/images (5).jpeg',
        './images1/images (6).jpeg',
        './images1/images (7).jpeg',
        './images1/images (8).jpeg',
        './images1/images (9).jpeg',
        './images1/images (10).jpeg',
        './images1/images.jpeg',
        './images1/images (11).jpeg',
        './images1/download (1).jpeg',
        './images1/download (2).jpeg',
        './images1/download (3).jpeg',
        './images1/download.jpeg',

    ];
  
    // Function to shuffle the array randomly
    function shuffleArray(array) {
      for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
      }
      return array;
    }
  
    // Function to create and display images randomly
    function displayRandomImages() {
      const shuffledImages = shuffleArray(imageUrls);
      // Clear the image container
      imageContainer.innerHTML = '';
      // Append shuffled images to the container
      shuffledImages.forEach((imageUrl, index) => {
        const img = document.createElement('img');
        img.src = imageUrl;
        img.alt = 'Random Image';
        const div = document.createElement('div');
        div.classList.add('flex-item');
        if (index === shuffledImages.length - 1) {
          div.classList.add('large-image'); // Make the last image larger
        }
        div.appendChild(img);
        imageContainer.appendChild(div);
      });
    }
  
    // Call the function to display random images initially
    displayRandomImages();
  });
  