var dropdownButtons = document.getElementsByClassName("dropbtn");

for (let i = 0; i < dropdownButtons.length; i++) {
  dropdownButtons[i].addEventListener('click', function(event) {
    event.stopPropagation();
    let parent = event.target.parentElement;
    let content = parent.children[1];
    console.log(content);
    if (content.style.display === 'block') {
      content.style.display = 'none';
    } else {
      content.style.display = 'block';
    }
  });
}