import { motion } from "framer-motion";
import { Check } from "lucide-react";

const plans = [
  {
    name: "Free Trial",
    price: "$0",
    period: "7 days",
    features: [
      "5 min of dubbing",
      "3 languages",
      "720p output",
      "Watermark",
    ],
    highlighted: false,
  },
  {
    name: "Pro",
    price: "$49",
    period: "/month",
    features: [
      "120 min of dubbing",
      "All 50+ languages",
      "4K output",
      "Voice cloning",
      "Priority support",
    ],
    highlighted: true,
  },
  {
    name: "Enterprise",
    price: "Custom",
    period: "",
    features: [
      "Unlimited dubbing",
      "Custom voices",
      "API access",
      "Dedicated manager",
      "SLA guarantee",
    ],
    highlighted: false,
  },
];

export default function Pricing() {
  return (
    <section
      id="pricing"
      className="relative py-32 bg-gradient-to-b from-black via-neutral-950 to-black"
    >
      <div className="max-w-6xl mx-auto px-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-20"
        >
          <h2 className="text-4xl sm:text-5xl font-bold text-white mb-4">
            Simple{" "}
            <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
              Pricing
            </span>
          </h2>
          <p className="text-neutral-400 text-lg">
            Start free, scale as you grow.
          </p>
        </motion.div>

        {/* Pricing Cards */}
        <div className="grid gap-8 sm:grid-cols-3">
          {plans.map((plan, i) => (
            <motion.div
              key={plan.name}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.12 }}
              className={`relative rounded-2xl p-8 flex flex-col bg-white/5 backdrop-blur border transition-all ${
                plan.highlighted
                  ? "border-cyan-400/50 shadow-[0_0_60px_rgba(34,211,238,0.25)]"
                  : "border-cyan-400/20 shadow-[0_0_40px_rgba(34,211,238,0.08)]"
              }`}
            >
              {/* Badge */}
              {plan.highlighted && (
                <span className="inline-flex w-fit mb-4 px-3 py-1 text-xs font-semibold tracking-wide rounded-full bg-cyan-400/10 text-cyan-400">
                  MOST POPULAR
                </span>
              )}

              <h3 className="text-xl font-semibold text-white mb-2">
                {plan.name}
              </h3>

              <div className="mb-6">
                <span className="text-4xl font-bold text-white">
                  {plan.price}
                </span>
                {plan.period && (
                  <span className="ml-1 text-neutral-400">
                    {plan.period}
                  </span>
                )}
              </div>

              <ul className="space-y-3 flex-1 mb-8">
                {plan.features.map((feature) => (
                  <li
                    key={feature}
                    className="flex items-center gap-3 text-sm text-neutral-300"
                  >
                    <Check className="w-4 h-4 text-cyan-400 flex-shrink-0" />
                    {feature}
                  </li>
                ))}
              </ul>

              <button
                className={`mt-auto py-3 rounded-xl text-sm font-semibold transition ${
                  plan.highlighted
                    ? "bg-cyan-400 text-black hover:opacity-90"
                    : "border border-white/15 text-white hover:bg-white/10"
                }`}
              >
                {plan.name === "Enterprise"
                  ? "Contact Sales"
                  : "Get Started"}
              </button>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
