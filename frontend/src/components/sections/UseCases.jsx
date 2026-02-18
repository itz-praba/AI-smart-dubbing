import { motion } from "framer-motion";
import { Youtube, BookOpen, Megaphone, Film } from "lucide-react";

const cases = [
  {
    icon: Youtube,
    title: "YouTubers & Creators",
    desc: "Expand your audience globally with dubbed content in every language.",
  },
  {
    icon: BookOpen,
    title: "E-Learning Platforms",
    desc: "Make educational content accessible to learners worldwide.",
  },
  {
    icon: Megaphone,
    title: "Marketing & Ads",
    desc: "Localize campaigns at scale for every target market.",
  },
  {
    icon: Film,
    title: "Film & Media Studios",
    desc: "Professional dubbing for movies, series, and documentaries.",
  },
];

export default function UseCases() {
  return (
    <section className="relative py-32 bg-gradient-to-b from-black via-neutral-950 to-black">
      <div className="max-w-6xl mx-auto px-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-20"
        >
          <h2 className="text-4xl sm:text-5xl font-bold text-white mb-4">
            Built for{" "}
            <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
              Every Creator
            </span>
          </h2>
        </motion.div>

        {/* Use case cards */}
        <div className="grid gap-8 sm:grid-cols-2 lg:grid-cols-4">
          {cases.map((c, i) => (
            <motion.div
              key={c.title}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.12 }}
              className="p-6 rounded-2xl text-center bg-white/5 backdrop-blur border border-cyan-400/20 shadow-[0_0_40px_rgba(34,211,238,0.08)] hover:border-cyan-400/40 transition-all"
            >
              <div className="w-12 h-12 rounded-xl bg-cyan-500/10 flex items-center justify-center mx-auto mb-5">
                <c.icon className="w-6 h-6 text-cyan-400" />
              </div>

              <h3 className="font-semibold text-white mb-2">
                {c.title}
              </h3>
              <p className="text-sm text-neutral-400 leading-relaxed">
                {c.desc}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
