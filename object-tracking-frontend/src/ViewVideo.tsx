
import React, { useState } from "react";
import "./index.css";

const ViewVideo: React.FC = () => {
    const [fileId, setFileId] = useState("");
    const [videoUrl, setVideoUrl] = useState<string | null>(null);
    const [error, setError] = useState<string>("");

    const fetchVideo = async () => {
        setError("");
        try {
            const response = await fetch(`http://localhost:8000/get-video/${fileId}`);
            if (!response.ok) {
                throw new Error("File not found");
            }
            const data = await response.json();
            setVideoUrl(data.s3_url);
        } catch (error) {
            setError("File not found or server error");
            setVideoUrl(null);
        }
    };

    return (
        <div className="container">
            <h2>View Processed Video</h2>
            <div className="input-group">
                <input
                    type="text"
                    placeholder="Enter File ID"
                    value={fileId}
                    onChange={(e) => setFileId(e.target.value)}
                    
                    
                />
                
                <button onClick={fetchVideo}>Fetch</button>
                <div style={{ marginTop: "5px" }}></div>
            </div>

            {error && <p className="error">{error}</p>}

            {videoUrl && (
                <div className="video-container">
                    <video controls className="video-player">
                        <source src={videoUrl} type="video/mp4" />
                        Your browser does not support the video tag.
                    </video>
                </div>
            )}
        </div>
    );
};

export default ViewVideo;
