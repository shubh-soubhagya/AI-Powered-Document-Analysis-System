document.addEventListener("DOMContentLoaded", function () {
    loadPlagiarismData();
    loadPDFList();
});

// Function to Upload Directory
function uploadDirectory() {
    let files = document.getElementById("directory-upload").files;
    if (files.length === 0) {
        alert("Please select a directory containing PDFs!");
        return;
    }

    let formData = new FormData();
    for (let file of files) {
        formData.append("pdfs", file);
    }

    fetch("/api/upload-directory", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
        loadPlagiarismData();
        loadPDFList();
    })
    .catch(error => console.error("Upload failed:", error));
}

// Function to Load Plagiarism Report
function loadPlagiarismData() {
    fetch("/api/plagiarism-report")
        .then(response => response.json())
        .then(data => {
            const tableBody = document.querySelector("#plagiarism-table tbody");
            tableBody.innerHTML = "";

            data.forEach(item => {
                let row = `
                    <tr>
                        <td>${item["File 1"]}</td>
                        <td>${item["File 2"]}</td>
                        <td>${item["Cosine (TF-IDF)"]}</td>
                        <td>${item["Jaccard"]}</td>
                        <td>${item["N-Gram"]}</td>
                    </tr>`;
                tableBody.innerHTML += row;
            });
        })
        .catch(error => console.error("Error fetching plagiarism data:", error));
}

// Function to Load PDF List
function loadPDFList() {
    fetch("/api/pdf-list")
        .then(response => response.json())
        .then(pdfs => {
            const pdfList = document.getElementById("pdf-list");
            pdfList.innerHTML = "";

            pdfs.forEach(pdf => {
                let listItem = document.createElement("li");
                listItem.innerHTML = `${pdf} <button onclick="openChatbot('${pdf}')">Ask AI</button>`;
                pdfList.appendChild(listItem);
            });
        })
        .catch(error => console.error("Error fetching PDFs:", error));
}

// Function to Open Chatbot for a Specific PDF
function openChatbot(pdfName) {
    const chatbox = document.getElementById("chatbox");
    chatbox.innerHTML = `<p>Chatbot for <strong>${pdfName}</strong> activated. Ask your queries below:</p>
                         <input type="text" id="query" placeholder="Type your question...">
                         <button onclick="sendQuery('${pdfName}')">Send</button>
                         <div id="chat-history"></div>`;
}
