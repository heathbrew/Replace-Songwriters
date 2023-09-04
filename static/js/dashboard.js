var button = document.getElementById("toggleButton");
var isOn = false;

button.addEventListener("click", function() {
  isOn = !isOn;
  if (isOn) {
    button.textContent = "ON";
    button.classList.add("on");
  } else {
    button.textContent = "OFF";
    button.classList.remove("on");
  }
});