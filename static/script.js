document.getElementById("uploadForm").addEventListener("submit", async function (event) {
    event.preventDefault();
    
    const fileInput = document.getElementById("pdfFile");
    if (fileInput.files.length === 0) {
        alert("Please select a PDF file.");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    document.getElementById("loader").classList.remove("hidden");
    document.getElementById("result").innerHTML = "";

    try {
        const response = await fetch("http://127.0.0.1:5000/analyze", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        document.getElementById("loader").classList.add("hidden");

        if (response.ok) {
            document.getElementById("result").innerHTML = `
                <h3>Analysis Results</h3>
                <p><strong>Majority Category:</strong> ${data.majority_label}</p>
                <p><strong>Predictions:</strong> ${data.all_predictions.join(", ")}</p>
            `;
        } else {
            document.getElementById("result").innerHTML = `<p style="color:red;">Error: ${data.error}</p>`;
        }
    } catch (error) {
        document.getElementById("result").innerHTML = `<p style="color:red;">Failed to connect to the server.</p>`;
        document.getElementById("loader").classList.add("hidden");
    }
});
