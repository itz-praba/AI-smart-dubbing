import { Upload, Video } from "lucide-react";
import { useRef, useState } from "react";

export default function StartDubbing() {
  const fileInputRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState(null);

  const handleUploadClick = () => {
    fileInputRef.current.click();
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFile(e.target.files[0]);
    }
  };

  return (
    <div className="min-h-screen pt-28 pb-28 bg-gradient-to-b from-black via-neutral-950 to-black px-6">
      
      {/* Glow background */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-cyan-500/10 blur-3xl rounded-full" />
      </div>

      <div className="max-w-3xl mx-auto">
        {/* Card */}
        <div className="relative w-full bg-white/5 backdrop-blur-xl border border-cyan-400/20 rounded-2xl p-12 shadow-2xl space-y-10">
          
          {/* Header */}
          <div className="text-center space-y-3">
            <h1 className="text-4xl sm:text-5xl font-bold text-white">
              Start <span className="text-cyan-400">Dubbing</span>
            </h1>
            <p className="text-neutral-400 max-w-xl mx-auto">
              Upload your video, choose a target language, and let AI deliver
              studio-quality dubbing in minutes.
            </p>
          </div>

          {/* Upload Box */}
          <div
            onClick={handleUploadClick}
            className="group relative border-2 border-dashed border-cyan-400/40 rounded-2xl p-12 text-center cursor-pointer
              transition hover:border-cyan-400 hover:bg-cyan-400/5"
          >
            {/* Hidden file input */}
            <input
              ref={fileInputRef}
              type="file"
              accept="video/mp4,video/mov,video/avi"
              onChange={handleFileChange}
              className="hidden"
            />

            <div className="flex flex-col items-center gap-4 text-neutral-400">
              <div className="w-14 h-14 rounded-xl bg-cyan-400/10 flex items-center justify-center group-hover:bg-cyan-400/20 transition">
                <Upload className="w-6 h-6 text-cyan-400" />
              </div>

              <p className="text-lg font-medium">
                {selectedFile ? "Video selected" : "Drag & drop your video here"}
              </p>

              <p className="text-sm text-neutral-500">
                {selectedFile ? selectedFile.name : "MP4, MOV, AVI • Max 2GB"}
              </p>

              <button
                type="button"
                onClick={handleUploadClick}
                className="mt-2 px-4 py-2 text-xs font-semibold rounded-lg
                  bg-cyan-400/10 text-cyan-400 hover:bg-cyan-400/20 transition"
              >
                Browse File
              </button>
            </div>
          </div>

          {/* Selected File Preview */}
          <div className="flex items-center gap-4 px-4 py-3 rounded-xl border border-white/10 bg-black/40">
            <Video className="w-5 h-5 text-cyan-400" />
            <p className="text-sm text-neutral-300">
              {selectedFile ? selectedFile.name : "No file selected"}
            </p>
          </div>

          {/* Language Select */}
          <div className="space-y-2">
            <label className="text-sm text-neutral-400">Target Language</label>

            <div className="relative">
              <select
                className="w-full appearance-none bg-black/60 border border-cyan-400/30 rounded-xl
                  px-4 py-3 pr-10 text-white focus:outline-none focus:border-cyan-400 transition"
              >
                <option value="">Select target language</option>
                <option>English</option>
                <option>Hindi</option>
                <option>Spanish</option>
                <option>French</option>
                <option>German</option>
                <option>Japanese</option>
              </select>

              <span className="pointer-events-none absolute inset-y-0 right-4 flex items-center text-cyan-400">
                ▼
              </span>
            </div>
          </div>

          {/* Start Button */}
          <button
            disabled={!selectedFile}
            className={`w-full py-4 rounded-xl font-semibold text-lg transition shadow-lg
              ${
                selectedFile
                  ? "bg-cyan-400 text-black hover:bg-cyan-300 shadow-cyan-400/30"
                  : "bg-cyan-400/40 text-black cursor-not-allowed"
              }`}
          >
            Start Dubbing
          </button>

          {/* Footer Hint */}
          <p className="text-center text-xs text-neutral-500">
            Your content is processed securely and never shared.
          </p>
        </div>
      </div>
    </div>
  );
}
