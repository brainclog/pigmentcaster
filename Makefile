
# Default target
.PHONY: run save
.DEFAULT_GOAL := run

# Run the Swift program
run:
	swift run -c release MetalPigmentRenderer

# Save the image with timestamp
save:
	@mkdir -p outputs
	@timestamp=$$(date +"%Y%m%d_%H%M%S"); \
	cp paint_strokes.png outputs/paint_strokes_$$timestamp.png; \
	echo "Saved as outputs/paint_strokes_$$timestamp.png"
