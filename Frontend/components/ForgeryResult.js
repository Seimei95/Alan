import { useEffect, useState } from "react";

export default function ForgeryResult() {
  const [img, setImg] = useState(null);

  useEffect(() => {
    const handler = () => {
      const res = localStorage.getItem("forgeryResult");
      setImg(res);
    };

    handler(); // load existing
    window.addEventListener("forgery-done", handler);

    return () => window.removeEventListener("forgery-done", handler);
  }, []);

  return (
    <div className="my-6">
      {img ? (
        <>
          <h2 className="text-xl font-semibold mb-2">Detected Forgery:</h2>
          <img src={img} alt="Forgery Result" className="border rounded shadow" />
        </>
      ) : (
        <p className="text-gray-600">No result yet. Upload an image to begin.</p>
      )}
    </div>
  );
}
