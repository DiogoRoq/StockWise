<template>
  <div class="file-upload-container">
    <h2>{{ title }}</h2>
    <div class="file-input-container">
      <label class="file-input-label">
        <span class="browse-button">Browse</span>
        <input 
          type="file"
          accept=".csv"
          class="file-input"
          @change="handleFileUpload"
        >
      </label>
      <p
        v-if="file" 
        class="file-name"
      >
        {{ file.name }}
      </p>
    </div>
    <button
      class="upload-button"
      @click="submitFile" 
    >
      Upload
    </button>

    <!-- Display success/error messages -->
    <div 
      v-if="message"
      :class="['message', isError ? 'error-message' : 'success-message']"
    >
      {{ message }}
    </div>

    <!-- Display RMSE and prediction message if available -->
    <div
      v-if="rmse !== null" 
      class="results-container"
    >
      <p class="rmse-message">
        Model trained successfully with RMSE: <strong>{{ rmse.toFixed(2) }}</strong>
      </p>
      <p class="prediction-message">
        You can now make predictions using the "Upload Prediction Data" section.
      </p>
    </div>

    <!-- Display predictions if available -->
    <div 
      v-if="predictions.length > 0" 
      class="predictions-container"
    >
      <h3>Predictions</h3>
      <p class="timeframe-message">
        The predicted demand is for <strong>weekly</strong> time frames.
      </p>
      <table>
        <thead>
          <tr>
            <th>SKU ID</th>
            <th>Predicted Demand</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="(prediction, index) in predictions" 
            :key="index"
          >
            <td>{{ prediction.sku_id }}</td>
            <td>{{ prediction['Predicted Demand'] }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  props: {
    title: {
      type: String,
      required: true,
    },
    endpoint: {
      type: String,
      required: true,
    },
  },
  data() {
    return {
      file: null, // Stores the selected file
      message: '', // Stores success or error messages
      isError: false, // Tracks whether the message is an error
      rmse: null, // Stores the RMSE value
      predictions: [], // Stores the prediction results
    };
  },
  methods: {
    // Handles file selection
    handleFileUpload(event) {
      this.file = event.target.files[0];
      this.message = ''; // Clear previous messages
      this.isError = false;
      this.rmse = null; // Clear previous RMSE
      this.predictions = []; // Clear previous predictions
    },
    // Submits the file to the backend
    async submitFile() {
      if (!this.file) {
        this.message = 'Please select a file.';
        this.isError = true;
        return;
      }

      const formData = new FormData();
      formData.append('file', this.file);

      try {
        // Dynamically determine the backend URL
        const backendURL = `${window.location.protocol}//${window.location.hostname}:8000`;

        // Send the file to the backend
        const uploadResponse = await axios.post(`${backendURL}${this.endpoint}`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });

        if (this.endpoint === "/upload/") {
          // Handle training response
          this.message = uploadResponse.data.message; // Display success message
          this.isError = false;

          // Poll the training status endpoint
          const statusEndpoint = `${backendURL}/training-status/`;
          const checkStatus = async () => {
            try {
              const statusResponse = await axios.get(statusEndpoint);
              if (statusResponse.data.status === "success") {
                this.rmse = statusResponse.data.rmse; // Display RMSE
              } else if (statusResponse.data.status === "error") {
                this.message = 'Training failed: ' + statusResponse.data.message;
                this.isError = true;
              } else if (statusResponse.data.status === "training") {
                // Training is still in progress, check again after a delay
                setTimeout(checkStatus, 1000); // Poll every 1 second
              }
            } catch (error) {
              this.message = 'Error checking training status: ' + (error.response?.data?.detail || 'Unknown error');
              this.isError = true;
            }
          };

          // Start polling
          checkStatus();
        } else if (this.endpoint === "/predict/") {
          // Handle prediction response
          this.message = "Predictions successful!"; // Display success message
          this.isError = false;
          this.predictions = uploadResponse.data.predictions; // Store predictions
        }
      } catch (error) {
        // Handle errors from the backend
        this.message = 'Error uploading file: ' + (error.response?.data?.detail || 'Unknown error');
        this.isError = true;
      }
    },
  },
};
</script>

<style scoped>
.file-upload-container {
  margin: 20px auto; /* Center the container */
  padding: 20px;
  border: 1px solid #03624d; /* Deep teal border */
  border-radius: 19px;
  background-color: #0E9273; /* Very dark teal background */
  max-width: 500px; /* Match the width of other elements */
  text-align: center; /* Center the content inside the container */
  color: #00df82; /* Vibrant green text on very dark teal */
}

h2 {
  color: #072828; /* White for headings */
  margin-bottom: 20px;
  padding-bottom: 15px;
}

.file-input-container {
  display: flex;
  flex-direction: column;
  align-items: center; /* Center the file input and file name */
  margin-bottom: 20px;
}

.file-input-label {
  display: inline-block;
  padding: 10px 20px;
  background-color: #072828; /* Vibrant green button */
  color: #00df82; /* Very dark teal text on vibrant green */
  border-radius: 5px;
  cursor: pointer;
  font-size: 16px;
  transition: background-color 0.3s ease;
}

.file-input-label:hover {
  background-color: #0D4949;
  color:#00df82; /* Slightly darker green on hover */
}

.browse-button {
  pointer-events: none; /* Ensure the label text is clickable */
}

.file-input {
  display: none; /* Hide the default file input */
}

.file-name {
  margin-top: 10px;
  color: #072828; /* Vibrant green text on very dark teal */
  font-size: 14px;
}

.upload-button {
  font-family: 'Poppins', sans-serif;
  padding: 10px 20px;
  background-color: #072828; /* Vibrant green button */
  color: #00df82; /* Very dark teal text on vibrant green */
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 16px;
  display: block; /* Make the button a block element */
  margin: 0 auto; /* Center the button */
}

.upload-button:hover {
  background-color: #0D4949;
  color:#00df82; /* Slightly darker green on hover */
}

.message {
  margin-top: 20px;
  padding: 10px;
  border-radius: 4px;
  font-size: 14px;
}

.success-message {
  background-color: #03624c; /* Deep teal background */
  color: #f0f0f0; /* Slightly darker white text on deep teal */
  border: 1px solid #03624c; /* Deep teal border */
}

.error-message {
  background-color: #3a1a1a; /* Dark red background */
  color: #f0a8a8; /* Light red text */
  border: 1px solid #6a2a2a; /* Dark red border */
}

.results-container {
  margin-top: 20px;
  padding: 15px;
  background-color: #030f0f; /* Very dark teal background */
  border: 1px solid #03624c; /* Deep teal border */
  border-radius: 4px;
}

.rmse-message {
  font-size: 16px;
  color: #00df82; /* Vibrant green text on very dark teal */
  margin-bottom: 10px;
}

.prediction-message {
  font-size: 14px;
  color: #00df82; /* Vibrant green text on very dark teal */
  font-style: italic;
}

.predictions-container {
  margin-top: 50px; /* Added more space above the table */
  padding: 15px;
  background-color: #030f0f; /* Very dark teal background */
  border: 1px solid #03624c; /* Deep teal border */
  border-radius: 4px;
}

.predictions-container h3 {
  margin-bottom: 15px;
  color: #ffffff; /* White for headings */
}

.timeframe-message {
  font-size: 14px;
  color: #00df82; /* Vibrant green text on very dark teal */
  margin-bottom: 10px;
  font-style: italic;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin: 0 auto; /* Center the table */
}

th, td {
  padding: 10px;
  text-align: center; /* Center the text in cells */
  border-bottom: 1px solid #03624c; /* Deep teal border */
  color: #e0e0e0; /* Light text color */
}

th {
  background-color: #00df82; /* Vibrant green header */
  color: #030f0f; /* Very dark teal text on vibrant green */
}

tr:hover {
  background-color: #03624c; /* Deep teal hover background */
  color: #f0f0f0; /* Slightly darker white text on deep teal */
}
</style>