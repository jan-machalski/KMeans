# K-Means Clustering Program

This program implements the **K-Means clustering algorithm**, designed to efficiently handle large datasets with up to **50 million data points**, supporting up to **20 dimensions** and **2-20 clusters**.

---

## Features
- Handles **large-scale datasets** with millions of points.
- Supports **1 to 20 dimensions** per data point.
- Allows **2 to 20 clusters** for classification.
- Implements **both text-based and binary data formats** for flexibility and performance.
- Optimized for **fast loading** of large datasets using binary format.

---

## Data Format

### **Text Format** (Human-Readable, Recommended for Small Datasets)
- First line: **Three integers** separated by spaces, indicating:
  - Number of points (**N**)
  - Number of dimensions (**d**)
  - Number of clusters (**k**)
- Next **N** lines: Each line contains **d floating-point values**, separated by spaces, representing a single data point.

#### **Example**:
```
1000 3 5
1.2 3.4 5.6
7.8 9.0 1.2
...
```

**Note:** While convenient for testing with small datasets, text-based loading can be slow for large-scale data.

---

### **Binary Format** (Optimized for Large Datasets)
- First **12 bytes**: Three **binary integers** (4 bytes each) representing:
  - Number of points (**N**)
  - Number of dimensions (**d**)
  - Number of clusters (**k**)
- Next **N segments**: Each data point consists of **d floating-point values** (4 bytes each), stored in binary format.

This format ensures **significantly faster data loading**, making it mandatory for large-scale processing.

---

## Performance Considerations
- **Text format** is user-friendly but inefficient for large data.
- **Binary format** is highly optimized for fast processing.
- Designed to handle up to **50 million points** efficiently.

---

## Installation & Usage

To run the K-Means clustering program, use the following syntax:

```bash
./Kmeans <format> <mode> <input_file> <output_file>
```

### **Example Usage:**
```bash
./Kmeans bin cpu points_5mln_4d_5c.dat output.txt
```
- **bin** → Specifies binary format (alternatively, use `txt` for text format)
- **cpu** → Specifies processing mode (could be `gpu` if supported)
- **points_5mln_4d_5c.dat** → Input dataset file
- **output.txt** → Output file containing clustering results

---

## Algorithm Description

### **CPU**
Both the assignment step and the update step are performed in loops without any parallel operations.

### **GPU1**
- The assignment step is executed in a kernel, which is called separately for each point.
- The update step utilizes:
  - `thrust::sort_by_key` to sort points by their cluster assignments.
  - `thrust::reduce_by_key` to compute new centroid positions efficiently.

### **GPU2**
- The assignment step remains the same as in **GPU1**.
- The update step employs a **reduction by key** approach:
  - For each centroid, a function with its corresponding flag is executed.
  - Only the points assigned to the given centroid are considered in the computation.

---
