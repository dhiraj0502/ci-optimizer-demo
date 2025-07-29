const express = require("express");
const app = express();
const indexRoute = require("./routes/index");

app.use("/", indexRoute);

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
