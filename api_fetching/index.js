
url = "http://13.48.59.12/api/similarity";
input  = {
    title : "The Shawshank Redemption",
}
fetch(url {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify(input), 
})
.then(response => response.json())
.then(console.log(response))