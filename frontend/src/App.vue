<template>
  <div id="app">
    <h1>Demand Forecasting</h1>
    <FileUpload
      title="Upload Training Data" 
      endpoint="/upload/" 
    />
    <FileUpload 
      title="Upload Prediction Data"
      endpoint="/predict/"
    />
  </div>
</template>

<script>
import FileUpload from './components/FileUpload.vue';

export default {
  name: 'App',
  components: {
    FileUpload,
  },
};
</script>

<style>
/* Global dark background */
body {
  background-color: #072623; /* Dark background for the entire page */
  color: #00DF82; /* Light text color for the entire page */
  margin: 0;
  font-family: Avenir, Helvetica, Arial, sans-serif;
}

#app {
  font-family: 'Poppins', sans-serif;
  text-align: center;
  margin-top: 60px;
}
</style>
