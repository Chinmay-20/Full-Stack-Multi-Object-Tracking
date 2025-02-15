import React, { useState } from "react";
import axios from "axios";

const UploadComponent: React.FC = () => {
    const [file, setFile] = useState<File | null>(null);
    const [response, setResponse] = useState<any>(null);
    const [error, setError] = useState<string>("");

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files && event.target.files.length > 0) {
            setFile(event.target.files[0]);
            setResponse(null); // Reset response when a new file is selected
        }
    };

    const handleUpload = async () => {
        if (!file) {
            setError("Please select a file.");
            return;
        }

        setError(""); // Clear previous errors

        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await axios.post("http://localhost:8000/upload", formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });
            setResponse(res.data);
        } catch (err) {
            setError("Failed to upload. Please try again.");
        }
    };

    return (
        <div className="container">
            <h2>Upload File for Object Tracking</h2>
            <input type="file" onChange={handleFileChange} accept="image/*,video/*" />
            <button onClick={handleUpload}>Upload</button>

            {error && <p className="error">{error}</p>}

            {response && (
                <div className="stats">
                    <h3>Upload Status:</h3>
                    <pre>{JSON.stringify(response, null, 2)}</pre>
                </div>
            )}
        </div>
    );
};

export default UploadComponent;
