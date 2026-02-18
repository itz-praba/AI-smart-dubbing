import { motion } from "framer-motion";
import { Star } from "lucide-react";

const testimonials = [
  {
    name: "Sarah Chen",
    role: "YouTuber, 2M subs",
    quote:
      "DubAI doubled my international viewership in one month. The voice cloning is insanely accurate.",
  },
  {
    name: "Marco Rossi",
    role: "Head of Content, EduPlatform",
    quote:
      "We localized 200+ courses in 3 weeks. Previously it took us 6 months with voice actors.",
  },
  {
    name: "Priya Sharma",
    role: "Marketing Director",
    quote:
      "Our ad campaigns now launch in 15 languages simultaneously. ROI went through the roof.",
  },
];

const logos = ["Netflix", "Spotify", "Adobe", "Coursera", "HubSpot"];

export default function Testimonials() {
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
            Trusted by{" "}
            <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
              Thousands
            </span>
          </h2>
        </motion.div>

        {/* Testimonial cards */}
        <div className="grid gap-8 md:grid-cols-3 mb-20">
          {testimonials.map((t, i) => (
            <motion.div
              key={t.name}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.12 }}
              className="p-6 rounded-2xl bg-white/5 backdrop-blur border border-cyan-400/20 shadow-[0_0_40px_rgba(34,211,238,0.08)]"
            >
              {/* Stars */}
              <div className="flex gap-1 mb-4">
                {Array.from({ length: 5 }).map((_, j) => (
                  <Star
                    key={j}
                    className="w-4 h-4 fill-cyan-400 text-cyan-400"
                  />
                ))}
              </div>

              <p className="text-sm text-neutral-300 leading-relaxed mb-6">
                “{t.quote}”
              </p>

              <div>
                <p className="text-sm font-semibold text-white">
                  {t.name}
                </p>
                <p className="text-xs text-neutral-400">
                  {t.role}
                </p>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Trust logos */}
        <div className="flex flex-wrap items-center justify-center gap-10 opacity-40">
          {logos.map((logo) => (
            <span
              key={logo}
              className="text-lg font-semibold tracking-wide text-white"
            >
              {logo}
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}
