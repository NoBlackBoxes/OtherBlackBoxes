const container = document.querySelector(".container")

const showLogo = () => {
    container.innerHTML = "<img src='images/logo.png'/>"
}
  

document.addEventListener("DOMContentLoaded", showLogo)
  
if ("serviceWorker" in navigator) {
    window.addEventListener("load", function() {
        navigator.serviceWorker
        .register("/serviceWorker.js")
        .then(res => console.log("service worker registered"))
        .catch(err => console.log("service worker not registered", err))
    })
}
