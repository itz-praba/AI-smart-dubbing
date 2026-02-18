import { motion } from "framer-motion";

const languages = [
  { code: "US", name: "English" },
  { code: "ES", name: "Spanish" },
  { code: "FR", name: "French" },
  { code: "DE", name: "German" },
  { code: "JP", name: "Japanese" },
  { code: "KR", name: "Korean" },
  { code: "CN", name: "Chinese" },
  { code: "BR", name: "Portuguese" },
  { code: "IN", name: "Hindi" },
  { code: "SA", name: "Arabic" },
  { code: "IT", name: "Italian" },
  { code: "RU", name: "Russian" },
  { code: "TR", name: "Turkish" },
  { code: "NL", name: "Dutch" },
  { code: "PL", name: "Polish" },
  { code: "SE", name: "Swedish" },
];

export default function Languages() {
  return (
    <section
      id="languages"
      className="relative py-32 bg-gradient-to-b from-black via-neutral-950 to-black"
    >
      <div className="max-w-7xl mx-auto px-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-20"
        >
          <h2 className="text-4xl sm:text-5xl font-bold text-white mb-4">
            50+ Supported{" "}
            <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
              Languages
            </span>
          </h2>
          <p className="text-neutral-400 text-lg">
            Reach every audience on the planet.
          </p>
        </motion.div>

        {/* Language Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-6">
          {languages.map((lang, i) => (
            <motion.div
              key={lang.code}
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.04 }}
              className="flex flex-col items-center justify-center gap-2 rounded-2xl p-6 bg-white/5 backdrop-blur border border-cyan-400/20 shadow-[0_0_35px_rgba(34,211,238,0.08)] hover:border-cyan-400/40 transition-all"
            >
              <span className="text-white font-semibold tracking-wide">
                {lang.code}
              </span>
              <span className="text-xs text-neutral-400">
                {lang.name}
              </span>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
