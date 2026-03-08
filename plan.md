# HVSR Progressive Layer Stripping - GUI Reorganization Plan

## Executive Summary

This document outlines a comprehensive reorganization of the HVSR Progressive Layer Stripping GUI to provide:
- Simpler, more intuitive navigation
- Consolidated batch processing
- Enhanced figure control
- Dedicated HVF Forward Modeling with flexible soil profile input
- Smart Vp/density estimation from Poisson's ratio

---

## 1. Current State Analysis

### Current GUI Structure (8 Pages)
```
MainWindow
├── Complete Workflow      (workflow_page.py)
├── Layer Stripping        (strip_page.py)
├── HV Forward             (forward_page.py)
├── Post-processing        (postprocess_page.py)
├── Report Generation      (report_page.py)
├── Batch Processing       (batch_page.py)
├── Advanced Analysis      (analysis_page.py)
└── Settings               (settings_page.py)
```

### Issues Identified
1. **Too many navigation items** - 8 pages is overwhelming
2. **Redundant functionality** - Workflow and Batch overlap conceptually
3. **Limited soil profile input** - Only file-based input supported
4. **No Vp/density estimation** - User must provide all values
5. **Limited figure control** - Basic plot settings scattered across pages
6. **No manual data entry** - Cannot input profile directly in GUI

---

## 2. Proposed New GUI Structure

### Simplified 4-Tab Layout
```
MainWindow (Tabbed Interface)
├── 🏠 Home / Quick Start
│   ├── Single Profile Analysis (streamlined)
│   ├── Batch Processing (consolidated)
│   └── Recent Projects
│
├── 📊 Soil Profile Editor
│   ├── Manual Table Entry
│   ├── File Import (CSV, TXT, HVf format)
│   ├── Vp/Density Auto-Estimation
│   └── Profile Validation & Preview
│
├── ⚡ HV Forward Modeling
│   ├── Direct HVf.exe computation
│   ├── Frequency configuration
│   ├── Live HV curve preview
│   └── Export options
│
└── 🎨 Visualization Studio
    ├── Figure Gallery (all generated plots)
    ├── Plot Customization Panel
    ├── Export Settings (format, DPI, etc.)
    └── Publication-Ready Templates
```

---

## 3. Detailed Feature Specifications

### 3.1 Home / Quick Start Tab

#### Single Profile Analysis Card
```
┌─────────────────────────────────────────────────────────┐
│ 🚀 Single Profile Analysis                              │
├─────────────────────────────────────────────────────────┤
│ Model File:    [________________] [Browse]              │
│ Output Dir:    [________________] [Browse]              │
│                                                         │
│ Frequency Range: [0.2] - [20.0] Hz  Points: [71]       │
│                                                         │
│ [ ] Run Layer Stripping                                │
│ [ ] Generate Reports                                   │
│                                                         │
│              [▶ Run Analysis]                          │
└─────────────────────────────────────────────────────────┘
```

#### Batch Processing Card
```
┌─────────────────────────────────────────────────────────┐
│ 📁 Batch Processing                                     │
├─────────────────────────────────────────────────────────┤
│ Profiles Directory: [________________] [Browse]         │
│ File Pattern:       [*.txt          ]                   │
│ Output Directory:   [________________] [Browse]         │
│                                                         │
│ ┌─────────────────────────────────────────────────┐    │
│ │ profile_01.txt  ✓                               │    │
│ │ profile_02.txt  ✓                               │    │
│ │ profile_03.txt  ✓                               │    │
│ │ ...                                              │    │
│ └─────────────────────────────────────────────────┘    │
│                                                         │
│ [▶ Run Batch]  [📊 Generate Comparative Report]        │
└─────────────────────────────────────────────────────────┘
```

---

### 3.2 Soil Profile Editor Tab (NEW)

This is a **major new feature** providing flexible soil profile input.

#### Input Methods
1. **Manual Table Entry** - Editable table in GUI
2. **CSV Import** - Standard CSV with headers
3. **TXT Import** - Space/tab delimited
4. **HVf Format** - Native format (N + layer rows)

#### Table Widget Design
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 📋 Soil Profile Editor                                    [+ Add Layer]     │
├─────────────────────────────────────────────────────────────────────────────┤
│ Layer │ Thickness │ Vs (m/s) │ Vp (m/s) │ ν (Poisson) │ Density │ Actions  │
│       │    (m)    │          │          │             │ (kg/m³) │          │
├───────┼───────────┼──────────┼──────────┼─────────────┼─────────┼──────────┤
│   1   │   5.0     │   150    │  [Auto]  │   [0.35]    │  1800   │ [↑][↓][✕]│
│   2   │   10.0    │   250    │  [Auto]  │   [0.33]    │  1900   │ [↑][↓][✕]│
│   3   │   15.0    │   400    │  [Auto]  │   [0.30]    │  2000   │ [↑][↓][✕]│
│  HS   │   ∞       │   800    │  [Auto]  │   [0.25]    │  2200   │    [✕]   │
└───────┴───────────┴──────────┴──────────┴─────────────┴─────────┴──────────┘
│                                                                             │
│ ⚙️ Auto-Estimation:  [✓] Calculate Vp from ν    [✓] Suggest ν from Vs     │
│                                                                             │
│ [📂 Import CSV] [📂 Import TXT] [💾 Export HVf] [🔄 Clear]                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Vp-Vs-Poisson's Ratio Relationship

**Formula:**
```
Vp = Vs × √[(2(1-ν)) / (1-2ν)]
```

Where:
- `Vp` = P-wave velocity (m/s)
- `Vs` = S-wave velocity (m/s)  
- `ν` = Poisson's ratio (dimensionless)

**Inverse formula (Poisson's ratio from Vp/Vs):**
```
ν = (Vp/Vs)² - 2 / [2((Vp/Vs)² - 1)]
```

#### Typical Poisson's Ratio Values by Soil Type

| Soil Type | Vs Range (m/s) | Typical ν | Suggested ν |
|-----------|----------------|-----------|-------------|
| Soft Clay (saturated) | 80-150 | 0.45-0.50 | 0.48 |
| Medium Clay | 150-250 | 0.40-0.45 | 0.42 |
| Stiff Clay | 250-400 | 0.35-0.40 | 0.38 |
| Loose Sand | 100-200 | 0.30-0.35 | 0.32 |
| Medium Dense Sand | 200-350 | 0.28-0.33 | 0.30 |
| Dense Sand | 350-500 | 0.25-0.30 | 0.28 |
| Gravel | 300-600 | 0.25-0.30 | 0.27 |
| Weathered Rock | 500-1000 | 0.20-0.28 | 0.25 |
| Intact Rock | >1000 | 0.15-0.25 | 0.22 |

#### Auto-Suggestion Algorithm
```python
def suggest_poisson_ratio(vs: float) -> float:
    """Suggest Poisson's ratio based on Vs value."""
    if vs < 150:      # Soft clay/silt
        return 0.48
    elif vs < 250:    # Medium clay / loose sand
        return 0.40
    elif vs < 400:    # Stiff clay / medium sand
        return 0.33
    elif vs < 600:    # Dense sand / gravel
        return 0.28
    elif vs < 1000:   # Weathered rock
        return 0.25
    else:             # Intact rock
        return 0.22
```

#### Density Estimation (Optional)
```python
def suggest_density(vs: float) -> float:
    """Estimate density from Vs using empirical relation."""
    # Based on Gardner's relation and typical soil correlations
    if vs < 150:
        return 1700  # Soft soils
    elif vs < 300:
        return 1850  # Medium soils
    elif vs < 500:
        return 2000  # Dense soils
    elif vs < 1000:
        return 2200  # Weathered rock
    else:
        return 2500  # Rock
```

---

### 3.3 HV Forward Modeling Tab (Enhanced)

**Key improvements:**
- Accept profile from Soil Profile Editor directly
- Real-time HV curve preview
- Multiple profile comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ⚡ HV Forward Modeling                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Profile Source:  (○) From Editor  (○) From File  [Browse...]               │
│                                                                             │
│ ┌─────────────────────────────┐  ┌─────────────────────────────────────┐   │
│ │ Frequency Configuration     │  │ Profile Preview                     │   │
│ ├─────────────────────────────┤  ├─────────────────────────────────────┤   │
│ │ Min Freq: [0.2 ] Hz        │  │ Layer  Thk   Vs    Vp    Rho       │   │
│ │ Max Freq: [20.0] Hz        │  │   1    5.0   150   290   1800      │   │
│ │ Points:   [71  ]           │  │   2    10    250   480   1900      │   │
│ │                             │  │   HS   ∞     800   1500  2200      │   │
│ │ [✓] Adaptive scanning      │  └─────────────────────────────────────┘   │
│ │ [✓] Log frequency spacing  │                                            │
│ └─────────────────────────────┘                                            │
│                                                                             │
│ ┌───────────────────────────────────────────────────────────────────────┐  │
│ │                         HV Curve Preview                               │  │
│ │                                                                        │  │
│ │     10 ┤                    *                                         │  │
│ │        │                   ***                                        │  │
│ │      5 ┤                  *   *                                       │  │
│ │        │                 *     *                                      │  │
│ │      1 ┼────*************───────**************************            │  │
│ │        └──────────────────────────────────────────────────            │  │
│ │          0.1      1        10       100                               │  │
│ │                    Frequency (Hz)                                      │  │
│ │                                                                        │  │
│ │  Peak: f₀ = 2.45 Hz, A₀ = 8.3                                         │  │
│ └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│ [▶ Compute HV Curve]  [💾 Save CSV]  [📊 Add to Comparison]                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 3.4 Visualization Studio Tab (NEW)

**Central hub for all figure management and customization.**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 🎨 Visualization Studio                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ ┌─────────────────────┐  ┌──────────────────────────────────────────────┐  │
│ │ 📁 Figure Gallery   │  │ 🖼️ Preview                                   │  │
│ ├─────────────────────┤  │                                              │  │
│ │ ▸ HV Curves         │  │  ┌────────────────────────────────────────┐ │  │
│ │   ├ Step0_5-layer   │  │  │                                        │ │  │
│ │   ├ Step1_4-layer   │  │  │     [Selected Figure Preview]          │ │  │
│ │   └ Step2_3-layer   │  │  │                                        │ │  │
│ │ ▸ Vs Profiles       │  │  │                                        │ │  │
│ │ ▸ Overlay Plots     │  │  └────────────────────────────────────────┘ │  │
│ │ ▸ Waterfall         │  │                                              │  │
│ │ ▸ Publication       │  │  Filename: hv_curve_Step0.png               │  │
│ └─────────────────────┘  │  Size: 1200x600 px  |  DPI: 200             │  │
│                          └──────────────────────────────────────────────┘  │
│                                                                             │
│ ┌───────────────────────────────────────────────────────────────────────┐  │
│ │ ⚙️ Plot Customization                                                  │  │
│ ├───────────────────────────────────────────────────────────────────────┤  │
│ │                                                                        │  │
│ │ Axes Settings                    Style Settings                       │  │
│ │ ├ X-axis: [Log   ▼]             ├ Line Width:  [2.0   ]              │  │
│ │ ├ Y-axis: [Linear▼]             ├ Line Color:  [🔵 #2E86AB]          │  │
│ │ ├ X-min:  [0.1   ]              ├ Peak Color:  [🔴 #E63946]          │  │
│ │ ├ X-max:  [50    ]              ├ Fill Alpha:  [0.15  ]              │  │
│ │ ├ Y-min:  [auto  ]              └ Grid Alpha:  [0.3   ]              │  │
│ │ └ Y-max:  [auto  ]                                                    │  │
│ │                                                                        │  │
│ │ Figure Dimensions                Annotations                          │  │
│ │ ├ Width:   [12   ] in           ├ [✓] Show peak marker               │  │
│ │ ├ Height:  [6    ] in           ├ [✓] Show peak label                │  │
│ │ └ DPI:     [300  ]              ├ [✓] Show frequency bands           │  │
│ │                                 └ [✓] Show legend                    │  │
│ │                                                                        │  │
│ │ Smoothing                        Title & Labels                       │  │
│ │ ├ [✓] Enable                    ├ Title:  [HVSR Curve        ]       │  │
│ │ ├ Window: [9    ]               ├ X-label: [Frequency (Hz)   ]       │  │
│ │ └ Order:  [3    ]               └ Y-label: [H/V Amplitude    ]       │  │
│ │                                                                        │  │
│ └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│ [🔄 Apply Changes]  [💾 Save Figure]  [📋 Copy to Clipboard]               │
│                                                                             │
│ Export:  Format: [PNG ▼]  DPI: [300]  [📤 Export All Figures]              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Implementation Plan

### Phase 1: Core Infrastructure (Week 1-2)
1. **Create new module structure**
   ```
   hvstrip_progressive/
   ├── core/
   │   ├── soil_profile.py      # NEW: Profile data structures
   │   ├── velocity_utils.py    # NEW: Vp/Vs/nu conversions
   │   └── ... (existing)
   ├── gui/
   │   ├── pages/
   │   │   ├── home_page.py           # NEW: Consolidated home
   │   │   ├── profile_editor_page.py # NEW: Soil profile editor
   │   │   ├── forward_page.py        # ENHANCED
   │   │   └── visualization_page.py  # NEW: Figure studio
   │   └── widgets/
   │       ├── layer_table.py     # NEW: Editable layer table
   │       ├── hv_preview.py      # NEW: Live HV preview
   │       └── figure_gallery.py  # NEW: Figure browser
   ```

2. **Implement velocity utilities module**
   ```python
   # core/velocity_utils.py
   
   import numpy as np
   from dataclasses import dataclass
   from typing import Optional, Tuple
   
   @dataclass
   class VelocityConverter:
       """Convert between Vp, Vs, and Poisson's ratio."""
       
       @staticmethod
       def vp_from_vs_nu(vs: float, nu: float) -> float:
           """Calculate Vp from Vs and Poisson's ratio."""
           if nu >= 0.5 or nu < 0:
               raise ValueError(f"Invalid Poisson's ratio: {nu}")
           return vs * np.sqrt((2 * (1 - nu)) / (1 - 2 * nu))
       
       @staticmethod
       def nu_from_vp_vs(vp: float, vs: float) -> float:
           """Calculate Poisson's ratio from Vp and Vs."""
           ratio_sq = (vp / vs) ** 2
           return (ratio_sq - 2) / (2 * (ratio_sq - 1))
       
       @staticmethod
       def suggest_nu(vs: float) -> float:
           """Suggest Poisson's ratio based on Vs."""
           if vs < 150:
               return 0.48
           elif vs < 250:
               return 0.40
           elif vs < 400:
               return 0.33
           elif vs < 600:
               return 0.28
           elif vs < 1000:
               return 0.25
           else:
               return 0.22
       
       @staticmethod
       def suggest_density(vs: float) -> float:
           """Suggest density based on Vs."""
           if vs < 150:
               return 1700
           elif vs < 300:
               return 1850
           elif vs < 500:
               return 2000
           elif vs < 1000:
               return 2200
           else:
               return 2500
   ```

### Phase 2: Soil Profile Editor (Week 2-3)
1. **Create SoilProfile dataclass**
   ```python
   # core/soil_profile.py
   
   @dataclass
   class Layer:
       thickness: float
       vs: float
       vp: Optional[float] = None
       nu: Optional[float] = None
       density: float = 2000.0
       is_halfspace: bool = False
       
       def compute_vp(self) -> float:
           """Compute Vp if nu is provided."""
           if self.vp is not None:
               return self.vp
           if self.nu is not None:
               return VelocityConverter.vp_from_vs_nu(self.vs, self.nu)
           # Default: use suggested nu
           nu = VelocityConverter.suggest_nu(self.vs)
           return VelocityConverter.vp_from_vs_nu(self.vs, nu)
   
   @dataclass  
   class SoilProfile:
       layers: List[Layer]
       name: str = "Unnamed Profile"
       
       def to_hvf_format(self) -> str:
           """Export to HVf model format."""
           ...
       
       @classmethod
       def from_csv(cls, path: Path) -> 'SoilProfile':
           """Import from CSV file."""
           ...
       
       @classmethod
       def from_hvf(cls, path: Path) -> 'SoilProfile':
           """Import from HVf format."""
           ...
   ```

2. **Build editable table widget**
   - Use QTableWidget with custom delegates
   - Real-time Vp calculation when nu changes
   - Validation feedback (red borders for invalid values)

### Phase 3: Enhanced Forward Modeling (Week 3-4)
1. **Direct profile integration**
   - Accept SoilProfile object directly
   - Write temporary HVf file for computation
   
2. **Live preview widget**
   - Matplotlib canvas embedded in Qt
   - Auto-refresh on parameter change (debounced)
   
3. **Comparison mode**
   - Store multiple results
   - Overlay plots with legend

### Phase 4: Visualization Studio (Week 4-5)
1. **Figure gallery browser**
   - Scan output directories for images
   - Thumbnail generation
   - Quick preview on selection

2. **Plot customization panel**
   - Real-time matplotlib figure manipulation
   - Store custom presets
   - Apply to batch of figures

3. **Export functionality**
   - Multiple format support (PNG, PDF, SVG, EPS)
   - Batch export with naming template
   - Resolution/DPI control

### Phase 5: Integration & Testing (Week 5-6)
1. **Connect all components**
2. **User workflow testing**
3. **Documentation update**

---

## 5. File Structure Changes

### New Files to Create
```
hvstrip_progressive/
├── core/
│   ├── soil_profile.py           # Profile data structures & I/O
│   └── velocity_utils.py         # Vp/Vs/nu conversion utilities
├── gui/
│   ├── pages/
│   │   ├── home_page.py          # Consolidated home/batch
│   │   ├── profile_editor_page.py # Soil profile editor
│   │   └── visualization_page.py  # Figure studio
│   └── widgets/
│       ├── layer_table_widget.py  # Editable layer table
│       ├── hv_preview_widget.py   # Live HV curve preview
│       ├── figure_gallery.py      # Figure browser/gallery
│       └── plot_config_panel.py   # Plot customization panel
```

### Files to Modify
```
hvstrip_progressive/gui/
├── main_window.py      # Reduce to 4 main tabs
├── pages/
│   └── forward_page.py # Enhance with profile integration
```

### Files to Deprecate (Keep but Remove from Navigation)
```
hvstrip_progressive/gui/pages/
├── workflow_page.py     # Merged into home_page
├── batch_page.py        # Merged into home_page  
├── strip_page.py        # Merged into home_page workflow
├── postprocess_page.py  # Merged into visualization_page
├── report_page.py       # Merged into visualization_page
├── analysis_page.py     # Merged into visualization_page
```

---

## 6. UI/UX Guidelines

### Design Principles
1. **Progressive disclosure** - Show basic options first, advanced on demand
2. **Visual feedback** - Loading indicators, success/error states
3. **Consistency** - Same patterns across all tabs
4. **Accessibility** - Keyboard navigation, tooltips

### Color Scheme (QFluentWidgets)
- Primary: System accent color (Auto theme)
- Success: #2E86AB (blue)
- Warning: #F4A261 (orange)  
- Error: #E63946 (red)
- Neutral: System grays

### Typography
- Titles: 14pt bold
- Subtitles: 12pt semibold
- Body: 10pt regular
- Captions: 9pt regular

---

## 7. Dependencies

### Existing (No Change)
- PySide6
- qfluentwidgets
- numpy
- matplotlib
- scipy
- pandas

### New (Optional Enhancements)
- pyqtgraph (for faster live plotting, optional)

---

## 8. Migration Path

### For Existing Users
1. Old pages remain functional (deprecated but accessible via settings)
2. Automatic migration of settings to new structure
3. Data files remain compatible

### For New Users
1. Start with simplified 4-tab interface
2. Guided tutorial on first launch
3. Example profiles included

---

## 9. Success Metrics

1. **Reduced navigation clicks** - 50% fewer clicks for common workflows
2. **Faster profile entry** - Manual entry takes <2 minutes for 5-layer profile
3. **Figure control** - All plot parameters accessible from single panel
4. **Learning curve** - New users productive within 10 minutes

---

## 10. Timeline Summary

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Phase 1: Infrastructure | 2 weeks | New module structure, velocity utils |
| Phase 2: Profile Editor | 1 week | Layer table, CSV/TXT import, Vp estimation |
| Phase 3: Forward Modeling | 1 week | Profile integration, live preview |
| Phase 4: Visualization | 1 week | Gallery, customization, export |
| Phase 5: Integration | 1 week | Testing, documentation, polish |
| **Total** | **6 weeks** | Complete reorganized GUI |

---

## 11. Quick Reference: Vp-Vs-ν Formulas

### Primary Formula
$$V_p = V_s \times \sqrt{\frac{2(1-\nu)}{1-2\nu}}$$

### Inverse Formula  
$$\nu = \frac{(V_p/V_s)^2 - 2}{2[(V_p/V_s)^2 - 1]}$$

### Typical Values Table

| Soil Type | Vs (m/s) | ν | Vp/Vs | Vp (m/s) |
|-----------|----------|-----|-------|----------|
| Soft Clay | 100 | 0.48 | 4.79 | 479 |
| Medium Clay | 200 | 0.40 | 2.45 | 490 |
| Stiff Clay | 300 | 0.35 | 2.08 | 624 |
| Loose Sand | 150 | 0.32 | 1.92 | 288 |
| Dense Sand | 400 | 0.28 | 1.80 | 720 |
| Gravel | 500 | 0.27 | 1.77 | 885 |
| Weathered Rock | 700 | 0.25 | 1.73 | 1211 |
| Intact Rock | 1500 | 0.22 | 1.66 | 2490 |

---

## 12. Next Steps

1. **Review this plan** with stakeholders
2. **Prioritize features** - Which are must-have vs nice-to-have?
3. **Create detailed mockups** for each new page
4. **Begin Phase 1** implementation

---

*Document created: January 27, 2026*
*Author: HVSR-Diffuse Development Team*
