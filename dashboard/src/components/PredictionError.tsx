import { AreaChart, Area, XAxis, YAxis, ResponsiveContainer, ReferenceLine } from 'recharts'

interface PredictionErrorProps {
  history: { tick: number; error: number }[]
}

/** Prediction error signal monitor — styled as a clinical waveform display.
 *  Resembles real-time physiological monitoring (EKG/EEG chart recorder style)
 *  with amber/orange trace on dark clinical background. */
export default function PredictionError({ history }: PredictionErrorProps) {
  const data = history.slice(-200)
  const currentError = data.length > 0 ? data[data.length - 1].error : 0

  return (
    <div style={{
      width: '100%',
      height: 'calc(100% - 24px)',
      padding: '4px 8px 4px 4px',
      position: 'relative',
    }}>
      {/* Current value readout — top right, clinical numeric display */}
      <div style={{
        position: 'absolute',
        top: '6px',
        right: '12px',
        zIndex: 2,
        display: 'flex',
        alignItems: 'baseline',
        gap: '4px',
      }}>
        <span style={{
          fontFamily: 'var(--font-clinical)',
          fontSize: '18px',
          fontWeight: 500,
          color: '#e8943a',
          fontVariantNumeric: 'tabular-nums',
          textShadow: '0 0 12px rgba(232, 148, 58, 0.3)',
        }}>
          {currentError.toFixed(3)}
        </span>
        <span style={{
          fontFamily: 'var(--font-clinical)',
          fontSize: '8px',
          color: 'var(--text-tertiary)',
          letterSpacing: '1px',
        }}>
          PE
        </span>
      </div>

      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 20, right: 8, left: -20, bottom: 0 }}>
          <defs>
            <linearGradient id="peGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#e8943a" stopOpacity={0.20} />
              <stop offset="60%" stopColor="#e8943a" stopOpacity={0.05} />
              <stop offset="100%" stopColor="#e8943a" stopOpacity={0} />
            </linearGradient>
          </defs>

          {/* Subtle grid — calibration marks */}
          <XAxis
            dataKey="tick"
            tick={{ fontSize: 8, fill: '#4a5270', fontFamily: 'var(--font-clinical)' }}
            tickLine={false}
            axisLine={{ stroke: 'rgba(100, 120, 160, 0.08)' }}
            interval="preserveStartEnd"
          />
          <YAxis
            tick={{ fontSize: 8, fill: '#4a5270', fontFamily: 'var(--font-clinical)' }}
            tickLine={false}
            axisLine={{ stroke: 'rgba(100, 120, 160, 0.08)' }}
            domain={[0, 'auto']}
            width={35}
          />

          {/* Baseline reference — zero prediction error line */}
          <ReferenceLine y={0} stroke="rgba(100, 120, 160, 0.1)" strokeDasharray="2 4" />

          {/* Error trace — clinical waveform */}
          <Area
            type="monotone"
            dataKey="error"
            stroke="#e8943a"
            strokeWidth={1.5}
            fill="url(#peGradient)"
            animationDuration={0}
            dot={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
