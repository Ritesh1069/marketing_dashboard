import React from 'react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

const CHART_COLORS = {
  primary: '#6366f1',
  secondary: '#14b8a6',
  accent: '#f43f5e',
  blue: '#3b82f6',
  indigo: '#6366f1',
  purple: '#8b5cf6',
  pink: '#ec4899',
  teal: '#14b8a6',
};

const ContentEvaluation = ({ data }) => {
  if (!data) return null;

  const {
    email_content,
    social_content,
    research_content,
    email_metrics,
    social_metrics,
    research_metrics,
    image_metrics,
    generated_image
  } = data;

  const metricsData = [
    { name: 'Email', ...email_metrics },
    { name: 'Social', ...social_metrics },
    { name: 'Research', ...research_metrics }
  ];

  const imageMetricsData = [
    { name: 'Sharpness', value: image_metrics?.sharpness || 0 },
    { name: 'Color Balance', value: image_metrics?.color_balance || 0 },
    { name: 'Composition', value: image_metrics?.composition || 0 },
    { name: 'Clarity', value: image_metrics?.clarity || 0 }
  ];

  const channelMetrics = Object.keys(email_metrics || {}).map(metric => ({
    metric,
    Email: email_metrics[metric],
    Social: social_metrics[metric],
    Research: research_metrics[metric]
  }));

  return (
    <div className="dashboard">
      <div className="grid">
        {/* Generated Content Display */}
        <div className="card card-full">
          <h3 className="subtitle">Generated Content</h3>
          <div className="content-section">
            <h4>Email Content</h4>
            <div className="content-text">{email_content}</div>
            
            <h4>Social Media Content</h4>
            <div className="content-text">{social_content}</div>
            
            <h4>Research Content</h4>
            <div className="content-text">{research_content}</div>
          </div>
        </div>

        {/* Generated Image Display */}
        {generated_image && (
          <div className="card card-full">
            <h3 className="subtitle">Generated Marketing Image</h3>
            <div className="image-container">
              <img 
                src={`data:image/jpeg;base64,${generated_image}`}
                alt="Generated marketing content"
                className="generated-image"
              />
            </div>
          </div>
        )}

        {/* Metrics Comparison Chart */}
        <div className="card">
          <h3 className="subtitle">Channel Performance Comparison</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={channelMetrics}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="metric" />
              <YAxis />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#ffffff',
                  border: '1px solid #e2e8f0',
                  borderRadius: '0.375rem',
                  boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)'
                }}
              />
              <Legend />
              <Bar dataKey="Email" fill={CHART_COLORS.primary} />
              <Bar dataKey="Social" fill={CHART_COLORS.secondary} />
              <Bar dataKey="Research" fill={CHART_COLORS.accent} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Metrics Over Time */}
        <div className="card">
          <h3 className="subtitle">Metrics Trends</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={metricsData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#ffffff',
                  border: '1px solid #e2e8f0',
                  borderRadius: '0.375rem',
                  boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)'
                }}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="accuracy" 
                stroke={CHART_COLORS.blue}
                strokeWidth={2}
                dot={{ fill: CHART_COLORS.blue }}
              />
              <Line 
                type="monotone" 
                dataKey="fluency" 
                stroke={CHART_COLORS.indigo}
                strokeWidth={2}
                dot={{ fill: CHART_COLORS.indigo }}
              />
              <Line 
                type="monotone" 
                dataKey="engagement" 
                stroke={CHART_COLORS.purple}
                strokeWidth={2}
                dot={{ fill: CHART_COLORS.purple }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Image Quality Metrics */}
        <div className="card">
          <h3 className="subtitle">Image Quality Analysis</h3>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={imageMetricsData}>
              <PolarGrid stroke="#e2e8f0" />
              <PolarAngleAxis dataKey="name" />
              <PolarRadiusAxis angle={30} domain={[0, 100]} />
              <Radar
                name="Image Quality"
                dataKey="value"
                stroke={CHART_COLORS.teal}
                fill={CHART_COLORS.teal}
                fillOpacity={0.6}
              />
              <Legend />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#ffffff',
                  border: '1px solid #e2e8f0',
                  borderRadius: '0.375rem',
                  boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)'
                }}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Metrics Table */}
        <div className="card card-full">
          <h3 className="subtitle">Detailed Metrics</h3>
          <div className="table-container">
            <table className="metrics-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Email</th>
                  <th>Social</th>
                  <th>Research</th>
                </tr>
              </thead>
              <tbody>
                {Object.keys(email_metrics || {}).map(metric => (
                  <tr key={metric}>
                    <td>{metric}</td>
                    <td>{email_metrics[metric].toFixed(2)}</td>
                    <td>{social_metrics[metric].toFixed(2)}</td>
                    <td>{research_metrics[metric].toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ContentEvaluation; 