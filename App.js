import React, { useState, useRef } from "react";
import { View, Text, TouchableOpacity, ActivityIndicator, StyleSheet, Alert, ScrollView, Dimensions } from "react-native";
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImageManipulator from "expo-image-manipulator";
import axios from "axios";

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState("");
  const [detections, setDetections] = useState([]);
  const [defectCounts, setDefectCounts] = useState({});
  const [debugInfo, setDebugInfo] = useState(null);
  const [cameraLayout, setCameraLayout] = useState({ width: 0, height: 0 });
  const cameraRef = useRef(null);

  // Make sure to use the correct IP address for your Flask server
  const SERVER_URL = "http://192.168.1.12:5000";

  const testServerConnection = async () => {
    try {
      console.log("🔍 Testing server connection...");
      setLoading(true);
      const response = await axios.get(`${SERVER_URL}/`, { timeout: 10000 });
      console.log("✅ Server connection successful:", response.data);
      
      Alert.alert(
        "✅ Server Status", 
        `Server is running!\n\nSupported Classes:\n${response.data.supported_classes.join('\n• ')}\n\nModel Info:\nInput: ${JSON.stringify(response.data.model_input_shape)}\nOutput: ${JSON.stringify(response.data.model_output_shape)}`,
        [{ text: "OK" }]
      );
    } catch (error) {
      console.error("❌ Server connection failed:", error);
      Alert.alert(
        "❌ Server Connection Failed", 
        `Cannot connect to server at ${SERVER_URL}\n\nError: ${error.message}\n\nMake sure:\n1. Flask server is running\n2. IP address is correct\n3. Both devices are on same network\n4. Port 5000 is not blocked`,
        [{ text: "OK" }]
      );
    } finally {
      setLoading(false);
    }
  };

  const handleCaptureAndPredict = async () => {
    if (!cameraRef.current) {
      Alert.alert("Error", "Camera not ready");
      return;
    }

    setLoading(true);
    setPrediction("");
    setDetections([]);
    setDefectCounts({});
    setDebugInfo(null);

    try {
      console.log("📸 Taking picture...");
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.9,
        base64: false,
        skipProcessing: false,
      });

      console.log("🖼️ Photo taken:", photo.uri);
      console.log("📏 Original dimensions:", photo.width, "x", photo.height);

      // Resize to 640x640 to match YOLO model input
      const manipulated = await ImageManipulator.manipulateAsync(
        photo.uri,
        [{ resize: { width: 640, height: 640 } }],
        { 
          compress: 0.95, // Higher quality for better detection
          format: ImageManipulator.SaveFormat.JPEG 
        }
      );

      console.log("✂️ Image resized to:", manipulated.width, "x", manipulated.height);

      const formData = new FormData();
      formData.append("file", {
        uri: manipulated.uri,
        name: "fabric_defect_image.jpg",
        type: "image/jpeg",
      });

      console.log("🚀 Sending to server for analysis...");
      
      const response = await axios.post(`${SERVER_URL}/predict`, formData, {
        headers: { 
          "Content-Type": "multipart/form-data",
        },
        timeout: 60000, // 60 second timeout for processing
      });

      console.log("✅ Server response received:", response.data);
      
      const result = response.data;
      setPrediction(result.prediction);
      setDetections(result.detections || []);
      setDefectCounts(result.defect_counts || {});
      setDebugInfo(result.debug_info || null);
      
      // Create detailed alert message
      let alertMessage = `Analysis Result: ${result.prediction}`;
      
      if (result.total_defects > 0) {
        alertMessage += `\n\n🔍 Detected ${result.total_defects} defect(s):`;
        
        // Show defect counts by type
        Object.entries(result.defect_counts).forEach(([defectType, count]) => {
          alertMessage += `\n• ${defectType}: ${count}`;
        });
        
        // Show detailed detection info
        alertMessage += `\n\n📊 Detection Details:`;
        result.detections.forEach((detection, index) => {
          const [x1, y1, x2, y2] = detection.bbox;
          alertMessage += `\n${index + 1}. ${detection.class_name}`;
          alertMessage += `\n   Confidence: ${(detection.confidence * 100).toFixed(1)}%`;
          alertMessage += `\n   Location: (${Math.round(x1)}, ${Math.round(y1)}) to (${Math.round(x2)}, ${Math.round(y2)})`;
        });
      } else {
        alertMessage += `\n\n✅ No defects detected in the fabric`;
        if (result.confidence > 0) {
          alertMessage += `\nModel confidence: ${(result.confidence * 100).toFixed(1)}%`;
        }
        
        // Show debug info for troubleshooting
        if (result.debug_info && result.debug_info.detection_counts_by_threshold) {
          alertMessage += `\n\n🔧 Detection Analysis:`;
          Object.entries(result.debug_info.detection_counts_by_threshold).forEach(([threshold, count]) => {
            if (count > 0) {
              alertMessage += `\n  At ${threshold.replace('threshold_', '')} confidence: ${count} potential detections`;
            }
          });
        }
      }
      
      // Show result alert
      Alert.alert(
        result.total_defects > 0 ? "⚠️ Defects Found!" : "✅ Fabric Quality OK", 
        alertMessage,
        [
          { text: "OK" },
          { 
            text: "Technical Details", 
            onPress: () => showTechnicalInfo(result) 
          }
        ]
      );

    } catch (error) {
      console.error("❌ Analysis failed:", error);
      console.error("❌ Error response:", error.response?.data);
      
      let errorMessage = "Analysis failed. ";
      
      if (error.code === 'ECONNREFUSED') {
        errorMessage += "Cannot connect to the AI server. Please ensure the Flask server is running.";
      } else if (error.code === 'ENOTFOUND') {
        errorMessage += "Server not found. Please check the IP address in the app settings.";
      } else if (error.code === 'ECONNABORTED') {
        errorMessage += "Request timed out. The image analysis is taking too long.";
      } else if (error.response?.status === 500) {
        errorMessage += `Server error: ${error.response?.data?.error || 'Internal server error'}`;
      } else if (error.response?.status === 413) {
        errorMessage += "Image file is too large. Please try with a smaller image.";
      } else if (error.response?.data?.error) {
        errorMessage += error.response.data.error;
      } else {
        errorMessage += error.message || "Unknown error occurred";
      }
      
      Alert.alert("❌ Analysis Failed", errorMessage, [
        { text: "OK" },
        { text: "Test Connection", onPress: testServerConnection }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const showTechnicalInfo = (result) => {
    let techInfo = "🔧 Technical Analysis Details:\n\n";
    
    if (result.debug_info) {
      const debug = result.debug_info;
      techInfo += `Model Configuration:\n`;
      techInfo += `• Input Shape: ${JSON.stringify(debug.model_input_shape)}\n`;
      techInfo += `• Output Shape: ${JSON.stringify(debug.model_output_shape)}\n`;
      techInfo += `• Output Range: ${debug.output_min_max[0].toFixed(4)} to ${debug.output_min_max[1].toFixed(4)}\n\n`;
      
      techInfo += `Supported Defect Types:\n`;
      debug.class_names?.forEach((className, index) => {
        techInfo += `• ${index}: ${className}\n`;
      });
      
      if (debug.detection_counts_by_threshold) {
        techInfo += `\nDetection Sensitivity Analysis:\n`;
        Object.entries(debug.detection_counts_by_threshold).forEach(([threshold, count]) => {
          techInfo += `• ${threshold.replace('threshold_', '')} confidence: ${count} detections\n`;
        });
      }
    }
    
    Alert.alert("🔧 Technical Details", techInfo);
  };

  // Calculate scaling factors for bounding box positioning
  const getScalingFactors = () => {
    if (cameraLayout.width === 0 || cameraLayout.height === 0) {
      return { scaleX: 1, scaleY: 1 };
    }
    
    // Model processes 640x640 images, scale to camera view size
    const scaleX = cameraLayout.width / 640;
    const scaleY = cameraLayout.height / 640;
    
    return { scaleX, scaleY };
  };

  // Get color for defect type
  const getDefectColor = (defectType) => {
    const colors = {
      'Hole': '#FF0000',           // Red - Critical defect
      'Spot': '#FF8C00',           // Orange - Visible defect  
      'Thick-Missing yarn': '#FFD700', // Gold - Thread issue
      'Knot-slub': '#FF1493',      // Deep Pink - Texture defect
      'Defect': '#FF0000'          // Default red
    };
    return colors[defectType] || '#FF6B6B';
  };

    // Get emoji for defect type
  const getDefectEmoji = (defectType) => {
    const emojis = {
      'Hole': '🕳️',
      'Spot': '🔴', 
      'Thick-Missing yarn': '🧵',
      'Knot-slub': '🪢',
    };
    return emojis[defectType] || '❌';
  };

  return (
    <View style={styles.container}>
      {!permission?.granted ? (
        <TouchableOpacity onPress={requestPermission} style={styles.permissionButton}>
          <Text style={styles.permissionText}>Grant Camera Permission</Text>
        </TouchableOpacity>
      ) : (
        <View 
          style={styles.cameraWrapper}
          onLayout={(event) => {
            const { width, height } = event.nativeEvent.layout;
            setCameraLayout({ width, height });
          }}
        >
          <CameraView 
            ref={cameraRef} 
            style={styles.camera}
            facing="back"
          />
          {detections.map((detection, index) => {
            const [x1, y1, x2, y2] = detection.bbox;
            const { scaleX, scaleY } = getScalingFactors();
            const left = x1 * scaleX;
            const top = y1 * scaleY;
            const width = (x2 - x1) * scaleX;
            const height = (y2 - y1) * scaleY;

            return (
              <View 
                key={index}
                style={[
                  styles.boundingBox,
                  {
                    left,
                    top,
                    width,
                    height,
                    borderColor: getDefectColor(detection.class_name),
                  }
                ]}
              >
                <Text style={styles.boxLabel}>
                  {getDefectEmoji(detection.class_name)} {detection.class_name}
                </Text>
              </View>
            );
          })}
        </View>
      )}

      <View style={styles.controls}>
        <TouchableOpacity 
          style={styles.captureButton}
          onPress={handleCaptureAndPredict}
          disabled={loading}
        >
          <Text style={styles.buttonText}>{loading ? "Analyzing..." : "Capture & Analyze"}</Text>
        </TouchableOpacity>

        <TouchableOpacity 
          style={styles.testButton}
          onPress={testServerConnection}
        >
          <Text style={styles.testButtonText}>Test Server Connection</Text>
        </TouchableOpacity>
      </View>

      {loading && <ActivityIndicator size="large" color="#007AFF" style={{ marginTop: 10 }} />}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f8f8f8",
    justifyContent: "flex-start",
  },
  cameraWrapper: {
    flex: 5,
    position: "relative",
  },
  camera: {
    flex: 1,
  },
  boundingBox: {
    position: "absolute",
    borderWidth: 2,
    borderRadius: 4,
    padding: 2,
    zIndex: 10,
    backgroundColor: 'rgba(0,0,0,0.3)',
  },
  boxLabel: {
    color: "#fff",
    fontWeight: "bold",
    fontSize: 12,
  },
  controls: {
    flex: 2,
    paddingHorizontal: 20,
    paddingTop: 10,
  },
  captureButton: {
    backgroundColor: "#007AFF",
    padding: 15,
    borderRadius: 10,
    alignItems: "center",
    marginBottom: 10,
  },
  testButton: {
    borderColor: "#007AFF",
    borderWidth: 1.5,
    padding: 12,
    borderRadius: 10,
    alignItems: "center",
  },
  buttonText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "600",
  },
  testButtonText: {
    color: "#007AFF",
    fontSize: 14,
    fontWeight: "600",
  },
  permissionButton: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#007AFF",
    padding: 20,
    margin: 20,
    borderRadius: 10,
  },
  permissionText: {
    color: "#fff",
    fontSize: 18,
  }
});
