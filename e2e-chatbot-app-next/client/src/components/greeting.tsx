import { motion } from 'framer-motion';
import { BrainIcon, HistoryIcon, ZapIcon } from 'lucide-react';
import { useSession } from '@/contexts/SessionContext';
import { useAppConfig } from '@/contexts/AppConfigContext';

const features = [
  {
    icon: BrainIcon,
    title: 'Cross-session Memory',
    description:
      'I remember facts about you across conversations. Tell me your name, preferences, or anything useful — I\'ll recall it next time.',
    examples: ['Remember my name is Alex', 'What do you remember about me?'],
  },
  {
    icon: HistoryIcon,
    title: 'Persistent Chat History',
    description:
      'Every conversation is saved to Lakebase (PostgreSQL). Pick up exactly where you left off — your full history is in the sidebar.',
    examples: ['Continue our earlier discussion', 'Show me my past chats'],
  },
  {
    icon: ZapIcon,
    title: 'Live Tools',
    description:
      'I can look up the current time and date, and more tools can be connected via Unity Catalog or MCP servers.',
    examples: ['What time is it?', 'What\'s today\'s date?'],
  },
];

export const Greeting = () => {
  const { session } = useSession();
  const { chatHistoryEnabled } = useAppConfig();

  const rawName =
    session?.user?.preferredUsername ??
    session?.user?.name ??
    null;

  // Use only the first word so email-style usernames like "john.doe@..." become "john"
  const name = rawName?.split(/[@.\s]/)[0] ?? null;

  return (
    <div
      key="overview"
      className="mx-auto mt-4 flex size-full max-w-3xl flex-col justify-center gap-6 px-4 md:mt-10 md:px-8"
    >
      {/* Headline */}
      <div className="flex flex-col gap-1">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 10 }}
          transition={{ delay: 0.3 }}
          className="font-semibold text-xl md:text-2xl"
        >
          {name ? `Hello, ${name}!` : 'Hello there!'}
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 10 }}
          transition={{ delay: 0.4 }}
          className="text-muted-foreground text-base"
        >
          I'm a memory-enabled assistant. Here's what I can do for you:
        </motion.div>
      </div>

      {/* Feature cards */}
      <div className="grid gap-3 sm:grid-cols-3">
        {features.map((feature, i) => {
          const Icon = feature.icon;
          // Dim the history card when chat history is disabled
          const dimmed = feature.icon === HistoryIcon && !chatHistoryEnabled;
          return (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 16 }}
              transition={{ delay: 0.45 + i * 0.08 }}
              className={`flex flex-col gap-2 rounded-xl border bg-muted/40 p-4 ${dimmed ? 'opacity-50' : ''}`}
            >
              <div className="flex items-center gap-2">
                <Icon className="size-4 shrink-0 text-primary" />
                <span className="font-medium text-sm">{feature.title}</span>
                {feature.icon === BrainIcon && (
                  <span className="ml-auto rounded-full bg-primary/10 px-1.5 py-0.5 text-primary text-xs font-medium">
                    Active
                  </span>
                )}
                {dimmed && (
                  <span className="ml-auto rounded-full bg-muted px-1.5 py-0.5 text-muted-foreground text-xs">
                    Off
                  </span>
                )}
              </div>
              <p className="text-muted-foreground text-xs leading-relaxed">
                {feature.description}
              </p>
              <div className="mt-auto flex flex-col gap-1">
                {feature.examples.map((ex) => (
                  <span
                    key={ex}
                    className="rounded-md bg-background px-2 py-1 text-xs text-muted-foreground italic"
                  >
                    "{ex}"
                  </span>
                ))}
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Tip */}
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.75 }}
        className="text-muted-foreground text-xs"
      >
        Tip: Use <span className="font-mono bg-muted px-1 rounded">remember</span>,{' '}
        <span className="font-mono bg-muted px-1 rounded">recall</span>, and{' '}
        <span className="font-mono bg-muted px-1 rounded">forget</span> in your messages to manage what I know about you.
      </motion.p>
    </div>
  );
};
