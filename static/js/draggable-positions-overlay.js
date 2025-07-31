/**
 * Draggable Open Positions Module
 * Makes the Open Positions dropdown moveable anywhere on the screen
 */

class DraggablePositionsOverlay {
    constructor() {
        this.overlay = null;
        this.isDragging = false;
        this.currentX = 0;
        this.currentY = 0;
        this.initialX = 0;
        this.initialY = 0;
        this.xOffset = 0;
        this.yOffset = 0;
        
        this.init();
    }
    
    init() {
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setupDragFunctionality());
        } else {
            this.setupDragFunctionality();
        }
    }
    
    setupDragFunctionality() {
        this.overlay = document.getElementById('positions-overlay');
        
        if (!this.overlay) {
            console.log('⚠️ Open Positions overlay not found, retrying...');
            setTimeout(() => this.setupDragFunctionality(), 500);
            return;
        }
        
        console.log('🎯 Setting up draggable Open Positions overlay');
        
        // Make the overlay draggable
        this.makeDraggable();
        
        // Add visual indicators
        this.addDragIndicators();
        
        // Add position persistence
        this.loadSavedPosition();
        
        console.log('✅ Open Positions overlay is now draggable!');
    }
    
    makeDraggable() {
        // Make the overlay absolutely positioned if it isn't already
        this.overlay.style.position = 'fixed';
        this.overlay.style.zIndex = '9999';
        this.overlay.style.cursor = 'move';
        
        // Add drag handle visual
        const header = this.overlay.querySelector('.positions-header');
        if (header) {
            header.style.cursor = 'grab';
            header.title = 'Drag to move positions panel';
            
            // Add drag icon
            const dragIcon = document.createElement('i');
            dragIcon.className = 'fas fa-grip-horizontal';
            dragIcon.style.cssText = `
                margin-left: 8px;
                opacity: 0.6;
                font-size: 12px;
                color: var(--text-tertiary);
            `;
            header.appendChild(dragIcon);
        }
        
        // Add event listeners
        this.overlay.addEventListener('mousedown', this.dragStart.bind(this));
        document.addEventListener('mousemove', this.dragMove.bind(this));
        document.addEventListener('mouseup', this.dragEnd.bind(this));
        
        // Touch events for mobile
        this.overlay.addEventListener('touchstart', this.dragStart.bind(this));
        document.addEventListener('touchmove', this.dragMove.bind(this));
        document.addEventListener('touchend', this.dragEnd.bind(this));
    }
    
    addDragIndicators() {
        // Add a subtle glow effect when hovering
        const style = document.createElement('style');
        style.textContent = `
            #positions-overlay {
                transition: all 0.2s ease;
            }
            
            #positions-overlay:hover {
                box-shadow: 0 0 20px rgba(74, 144, 226, 0.3);
                transform: scale(1.02);
            }
            
            #positions-overlay.dragging {
                box-shadow: 0 0 30px rgba(74, 144, 226, 0.5);
                transform: rotate(2deg);
                opacity: 0.9;
                cursor: grabbing !important;
            }
            
            #positions-overlay.dragging .positions-header {
                cursor: grabbing !important;
            }
            
            .drag-indicator {
                position: absolute;
                top: -10px;
                left: 50%;
                transform: translateX(-50%);
                background: var(--primary-color);
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 10px;
                opacity: 0;
                transition: opacity 0.2s ease;
                pointer-events: none;
                z-index: 10000;
            }
            
            #positions-overlay:hover .drag-indicator {
                opacity: 1;
            }
        `;
        document.head.appendChild(style);
        
        // Add drag indicator
        const indicator = document.createElement('div');
        indicator.className = 'drag-indicator';
        indicator.textContent = 'Drag to move';
        this.overlay.appendChild(indicator);
    }
    
    dragStart(e) {
        // Only allow dragging from the header
        const header = this.overlay.querySelector('.positions-header');
        if (!header.contains(e.target) && e.target !== this.overlay) {
            return;
        }
        
        this.isDragging = true;
        this.overlay.classList.add('dragging');
        
        if (e.type === 'touchstart') {
            this.initialX = e.touches[0].clientX - this.xOffset;
            this.initialY = e.touches[0].clientY - this.yOffset;
        } else {
            this.initialX = e.clientX - this.xOffset;
            this.initialY = e.clientY - this.yOffset;
        }
        
        e.preventDefault();
    }
    
    dragMove(e) {
        if (!this.isDragging) return;
        
        e.preventDefault();
        
        if (e.type === 'touchmove') {
            this.currentX = e.touches[0].clientX - this.initialX;
            this.currentY = e.touches[0].clientY - this.initialY;
        } else {
            this.currentX = e.clientX - this.initialX;
            this.currentY = e.clientY - this.initialY;
        }
        
        this.xOffset = this.currentX;
        this.yOffset = this.currentY;
        
        // Constrain to viewport
        const rect = this.overlay.getBoundingClientRect();
        const maxX = window.innerWidth - rect.width;
        const maxY = window.innerHeight - rect.height;
        
        this.xOffset = Math.max(0, Math.min(maxX, this.xOffset));
        this.yOffset = Math.max(0, Math.min(maxY, this.yOffset));
        
        this.updatePosition();
    }
    
    dragEnd(e) {
        if (!this.isDragging) return;
        
        this.isDragging = false;
        this.overlay.classList.remove('dragging');
        
        // Save position
        this.savePosition();
        
        console.log(`📍 Positions panel moved to: ${this.xOffset}, ${this.yOffset}`);
    }
    
    updatePosition() {
        this.overlay.style.transform = `translate(${this.xOffset}px, ${this.yOffset}px)`;
    }
    
    savePosition() {
        const position = {
            x: this.xOffset,
            y: this.yOffset,
            timestamp: Date.now()
        };
        localStorage.setItem('goldgpt_positions_overlay_position', JSON.stringify(position));
    }
    
    loadSavedPosition() {
        try {
            const savedPosition = localStorage.getItem('goldgpt_positions_overlay_position');
            if (savedPosition) {
                const position = JSON.parse(savedPosition);
                this.xOffset = position.x || 0;
                this.yOffset = position.y || 0;
                this.updatePosition();
                console.log(`📍 Restored positions panel to: ${this.xOffset}, ${this.yOffset}`);
            }
        } catch (e) {
            console.log('⚠️ Could not restore positions panel position:', e);
        }
    }
    
    // Public methods for programmatic control
    resetPosition() {
        this.xOffset = 0;
        this.yOffset = 0;
        this.updatePosition();
        this.savePosition();
        console.log('🔄 Reset positions panel to original position');
    }
    
    centerPosition() {
        const rect = this.overlay.getBoundingClientRect();
        this.xOffset = (window.innerWidth - rect.width) / 2;
        this.yOffset = (window.innerHeight - rect.height) / 2;
        this.updatePosition();
        this.savePosition();
        console.log('🎯 Centered positions panel');
    }
    
    snapToCorner(corner = 'top-right') {
        const rect = this.overlay.getBoundingClientRect();
        const margin = 20;
        
        switch (corner) {
            case 'top-left':
                this.xOffset = margin;
                this.yOffset = margin;
                break;
            case 'top-right':
                this.xOffset = window.innerWidth - rect.width - margin;
                this.yOffset = margin;
                break;
            case 'bottom-left':
                this.xOffset = margin;
                this.yOffset = window.innerHeight - rect.height - margin;
                break;
            case 'bottom-right':
                this.xOffset = window.innerWidth - rect.width - margin;
                this.yOffset = window.innerHeight - rect.height - margin;
                break;
        }
        
        this.updatePosition();
        this.savePosition();
        console.log(`📍 Snapped positions panel to ${corner}`);
    }
}

// Global instance
window.draggablePositions = new DraggablePositionsOverlay();

// Add keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.shiftKey && window.draggablePositions) {
        switch (e.key) {
            case 'R':
                e.preventDefault();
                window.draggablePositions.resetPosition();
                break;
            case 'C':
                e.preventDefault();
                window.draggablePositions.centerPosition();
                break;
            case '1':
                e.preventDefault();
                window.draggablePositions.snapToCorner('top-left');
                break;
            case '2':
                e.preventDefault();
                window.draggablePositions.snapToCorner('top-right');
                break;
            case '3':
                e.preventDefault();
                window.draggablePositions.snapToCorner('bottom-left');
                break;
            case '4':
                e.preventDefault();
                window.draggablePositions.snapToCorner('bottom-right');
                break;
        }
    }
});

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DraggablePositionsOverlay;
}
