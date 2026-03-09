import { Upload, Video } from "lucide-react";
import { useRef, useState } from "react";
import axios from "axios";
import darkAlert from "../utils/sweetalert";

const VIDEO_BUCKET = import.meta.env.VITE_AWS_S3_BUCKET_NAME;

export default function StartDubbing() {
  const fileInputRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [sourceLanguage, setSourceLanguage] = useState("");
  const [targetLanguage, setTargetLanguage] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [dubbedVideoUrl, setDubbedVideoUrl] = useState(null);

  const handleUploadClick = () => {
    fileInputRef.current.click();
  };
  
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFile(e.target.files[0]);
    }
  };
  
  const languages = [
  { label: "English", value: "en" },
  { label: "Tamil", value: "ta" },
  { label: "Spanish", value: "es" },
  { label: "French", value: "fr" },
  { label: "German", value: "de" },
  { label: "Italian", value: "it" },
  { label: "Portuguese", value: "pt" },
  { label: "Dutch", value: "nl" },
  { label: "Russian", value: "ru" },
  { label: "Chinese (Simplified)", value: "zh-cn" },
  { label: "Japanese", value: "ja" },
  { label: "Korean", value: "ko" },
  { label: "Arabic", value: "ar" },
  { label: "Turkish", value: "tr" },
  { label: "Polish", value: "pl" },
  { label: "Vietnamese", value: "vi" },
  { label: "Ukrainian", value: "uk" },
  { label: "Czech", value: "cs" },
  { label: "Danish", value: "da" },
  { label: "Finnish", value: "fi" },
  { label: "Norwegian", value: "no" },
  { label: "Swedish", value: "sv" },
  { label: "Greek", value: "el" },
  { label: "Hebrew", value: "he" },
  { label: "Thai", value: "th" },
];

const handleStartDubbing = async () => {
  if (!selectedFile || !sourceLanguage || !targetLanguage) return;

  try {
    setIsProcessing(true);

    // STEP 1: Upload video using axios
    const formData = new FormData();
    formData.append("file", selectedFile);

    const uploadResponse = await axios.post(
      "http://localhost:8001/upload-video",
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      }
    );

    const uploadData = uploadResponse.data;

    // STEP 2: Call dub-video endpoint using axios
    const dubResponse = await axios.post(
      "http://localhost:8001/ai/dub-video",
      {
        video_bucket: VIDEO_BUCKET,
        video_key: uploadData.videoKey,
        target_language: targetLanguage,
        source_language: sourceLanguage,
        whisper_model_size: "medium",
        diarize: true,
        enable_lip_sync: true,
        video_codec: "copy",
        audio_codec: "aac",
        audio_bitrate: "192k",
        enable_audio_mastering: true, 
        enable_bgm_ducking: true, 
        target_lufs: -16.0, 
        target_true_peak: -1.5, 
        dialogue_peak_db: -3.0, 
        bgm_duck_db: -12.0, 
        comp_threshold_db: -24.0, 
        comp_ratio: 4.0, 
        comp_makeup_db: 6.0,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    );

    const dubData = dubResponse.data;

    const videoUrl = `https://${dubData.dubbed_video_bucket}.s3.ap-south-1.amazonaws.com/${dubData.dubbed_video_key}`;
    setDubbedVideoUrl(videoUrl);
  } catch (error) {
    console.error("Dubbing error:", error.response?.data || error.message);
    // alert("Something went wrong while dubbing.");
    // ❌ Error SweetAlert
      darkAlert.fire({
        icon: "error",
        title: "Failed",
        text: "Something went wrong while dubbing.",
      });
  } finally {
    setIsProcessing(false);
  }
};

  return (
    <div className="min-h-screen pt-28 pb-28 bg-gradient-to-b from-black via-neutral-950 to-black px-6 relative">
      
      {/* Processing Overlay */}
      {isProcessing && (
        <div className="absolute inset-0 bg-black/80 backdrop-blur-md z-50 flex flex-col items-center justify-center">
          <div className="w-16 h-16 border-4 border-cyan-400 border-t-transparent rounded-full animate-spin mb-6" />
          <p className="text-cyan-400 text-lg font-semibold">
            Processing your video...
          </p>
          <p className="text-neutral-400 text-sm mt-2">
            AI is generating studio-quality dubbing
          </p>
        </div>
      )}

      {/* Glow background */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-cyan-500/10 blur-3xl rounded-full" />
      </div>

      <div className="max-w-3xl mx-auto">
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

          {/* Selected File */}
          <div className="flex items-center gap-4 px-4 py-3 rounded-xl border border-white/10 bg-black/40">
            <Video className="w-5 h-5 text-cyan-400" />
            <p className="text-sm text-neutral-300">
              {selectedFile ? selectedFile.name : "No file selected"}
            </p>
          </div>

          {/* Source Language */}
          <div className="space-y-2">
            <label className="text-sm text-neutral-400">Video Language</label>
            <select
              value={sourceLanguage}
              onChange={(e) => setSourceLanguage(e.target.value)}
              className="w-full bg-black/60 border border-cyan-400/30 rounded-xl px-4 py-3 text-white"
            >
              <option value="">Select video language</option>
             {languages.map((lang) => (
                <option key={lang.value} value={lang.value}>
                  {lang.label}
                </option>
              ))}
            </select>
          </div>

          {/* Target Language */}
          <div className="space-y-2">
            <label className="text-sm text-neutral-400">Target Language</label>
            <select
              value={targetLanguage}
              onChange={(e) => setTargetLanguage(e.target.value)}
              className="w-full bg-black/60 border border-cyan-400/30 rounded-xl px-4 py-3 text-white"
            >
              <option value="">Select target language</option>
              {languages.map((lang) => (
                <option key={lang.value} value={lang.value}>
                  {lang.label}
                </option>
              ))}
            </select>
          </div>

          {/* Start Button */}
          <button
            disabled={!selectedFile || !sourceLanguage || !targetLanguage}
            onClick={handleStartDubbing}
            className={`w-full py-4 rounded-xl font-semibold text-lg transition shadow-lg
              ${
                selectedFile && sourceLanguage && targetLanguage
                  ? "bg-cyan-400 text-black hover:bg-cyan-300 shadow-cyan-400/30"
                  : "bg-cyan-400/40 text-black cursor-not-allowed"
              }`}
          >
            Start Dubbing
          </button>

          {/* Dubbed Video Output */}
          {dubbedVideoUrl && (
            <div className="space-y-4">
              <h2 className="text-white text-xl font-semibold">
                Dubbed Video Output
              </h2>
              <video
                controls
                src={dubbedVideoUrl}
                className="w-full rounded-xl border border-cyan-400/20"
              />
            </div>
          )}

          <p className="text-center text-xs text-neutral-500">
            Your content is processed securely and never shared.
          </p>
        </div>
      </div>
    </div>
  );
}