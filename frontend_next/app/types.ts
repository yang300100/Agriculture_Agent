export type JsonMap = Record<string, any>;

export type Session = {
  id: string;
  title: string;
  message_count: number;
  updated_at?: string;
};

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  image?: { name: string; url: string };
};

export type Profile = {
  user_region: string;
  user_soil_type: string;
  user_farm_size: number;
  user_experience: string;
  user_goals: string[];
  user_phone: string;
  autonomy_level?: "low" | "medium" | "high";
};

export type Field = {
  id: string;
  name: string;
  area_mu: number;
  area_m2: number;
  coordinates: number[][];
  center_lat: number;
  center_lon: number;
  soil_type: string;
  current_crop: string;
  history: JsonMap[];
};

export type Task = JsonMap & {
  id: string;
  title: string;
  crop: string;
  status: string;
  priority: string;
};

export type Progress = JsonMap & {
  id: string;
  crop: string;
  stage: string;
  progress_percent: number;
  status: string;
};

export type Device = JsonMap & {
  device_id: string;
  name: string;
  driver: string;
  capabilities: string[];
  sensors: string[];
  status: string;
  state: JsonMap;
  connection?: JsonMap;
  initial_state?: JsonMap;
  location?: string;
  plot_id?: string;
  zone_id?: string;
  plot_name?: string;
  editable?: boolean;
};

export type Rule = JsonMap & { id: string; name: string; enabled: boolean };
