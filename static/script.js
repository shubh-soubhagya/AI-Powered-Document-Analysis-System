// function uploadFile() {
//     let fileInput = document.getElementById('fileInput').files[0];

//     if (!fileInput) {
//         alert("Please select a file to upload.");
//         return;
//     }

//     let formData = new FormData();
//     formData.append("file", fileInput);

//     fetch("/analyze", {
//         method: "POST",
//         body: formData
//     })
//     .then(response => response.json())
//     .then(data => {
//         if (data.error) {
//             document.getElementById("result").innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
//         } else {
//             document.getElementById("result").innerHTML = `
//                 <h3>Analysis Results</h3>
//                 <p><strong>Category:</strong> ${data.majority_label}</p>
//                 <p><strong>Plagiarism Score:</strong> ${data.plagiarism_score}</p>
//             `;
//         }
//     })
//     .catch(error => console.error("Error:", error));
// }

// function askQuestion() {
//     let query = document.getElementById('queryInput').value.trim();

//     if (!query) {
//         alert("Please enter a question.");
//         return;
//     }

//     fetch("/qna", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ query: query })
//     })
//     .then(response => response.json())
//     .then(data => {
//         if (data.error) {
//             document.getElementById("qnaResult").innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
//         } else {
//             document.getElementById("qnaResult").innerHTML = `<p><strong>Answer:</strong> ${data.answer}</p>`;
//         }
//     })
//     .catch(error => console.error("Error:", error));
// }


document.getElementById('directoryForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const directoryPath = document.getElementById('directoryPath').value;
    const loadingOverlay = document.getElementById('loadingOverlay');
    const progressBar = document.getElementById('progressBar');
    const statusMessage = document.getElementById('statusMessage');
    const detailedStatus = document.getElementById('detailedStatus');
    
    // Show loading overlay
    loadingOverlay.style.display = 'flex';
    
    // Send directory path to server
    fetch('/upload', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: 'directory_path=' + encodeURIComponent(directoryPath)
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('Error: ' + data.error);
            loadingOverlay.style.display = 'none';
            return;
        }
        
        // Start checking status
        checkStatus();
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred. Please try again.');
        loadingOverlay.style.display = 'none';
    });
});

function checkStatus() {
    fetch('/status')
        .then(response => response.json())
        .then(data => {
            const progressBar = document.getElementById('progressBar');
            const detailedStatus = document.getElementById('detailedStatus');
            
            progressBar.style.width = data.progress + '%';
            detailedStatus.textContent = data.status_message;
            
            if (data.error) {
                alert('Error: ' + data.error);
                document.getElementById('loadingOverlay').style.display = 'none';
                return;
            }
            
            if (data.is_processing) {
                // Check again in 1 second
                setTimeout(checkStatus, 1000);
            } else if (data.progress === 100) {
                // Redirect to dashboard when complete
                window.location.href = '/dashboard';
            }
        })
        .catch(error => {
            console.error('Error checking status:', error);
            setTimeout(checkStatus, 2000); // Try again after 2 seconds
        });
}