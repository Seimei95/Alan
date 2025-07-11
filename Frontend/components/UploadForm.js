import { useState } from "react";
import axios from "axios";

export default function UploadForm() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) return alert("Please select an image");

    const formData = new FormData();
    formData.append("image", file);
    setLoading(true);

    try {
      const res = await axios.post("http://localhost:5000/upload", formData, {
        responseType: "blob",
      });

      const blobUrl = URL.createObjectURL(res.data);
      localStorage.setItem("forgeryResult", blobUrl);
      window.dispatchEvent(new Event("forgery-done"));
    } catch (err) {
      alert("Upload failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mb-4">
      <input
        type="file"
        accept="image/*"
        onChange={(e) => setFile(e.target.files[0])}
        className="block mb-2"
      />
      <button
        onClick={handleUpload}
        className="px-4 py-2 bg-blue-600 text-white rounded"
      >
        {loading ? "Detecting..." : "Upload & Detect"}
      </button>
    </div>
  );
}
