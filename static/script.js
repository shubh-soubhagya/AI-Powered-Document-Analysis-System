function uploadFile() {
    let fileInput = document.getElementById('fileInput').files[0];

    if (!fileInput) {
        alert("Please select a file to upload.");
        return;
    }

    let formData = new FormData();
    formData.append("file", fileInput);

    fetch("/analyze", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById("result").innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
        } else {
            document.getElementById("result").innerHTML = `
                <h3>Analysis Results</h3>
                <p><strong>Category:</strong> ${data.majority_label}</p>
                <p><strong>Plagiarism Score:</strong> ${data.plagiarism_score}</p>
            `;
        }
    })
    .catch(error => console.error("Error:", error));
}

function askQuestion() {
    let query = document.getElementById('queryInput').value.trim();

    if (!query) {
        alert("Please enter a question.");
        return;
    }

    fetch("/qna", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById("qnaResult").innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
        } else {
            document.getElementById("qnaResult").innerHTML = `<p><strong>Answer:</strong> ${data.answer}</p>`;
        }
    })
    .catch(error => console.error("Error:", error));
}
