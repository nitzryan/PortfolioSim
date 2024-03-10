// Select the button element
var dropdownButton = document.querySelector('.dropbtn');

dropdownButton.addEventListener('click', function() {
  var dropdownContent = document.getElementById('sources');
  if (dropdownContent.style.display === 'block') {
    dropdownContent.style.display = 'none';
  } else {
    dropdownContent.style.display = 'block';
  }
});