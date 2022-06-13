# public set
python3 test_seg.py --root ../dataset/ \
                    --load ./model_seg.pth \
                    --subject S5 \
                    --valid \
                    --model deeplab \
                    --conf_threshold 0.85 \
                    --val_threshold 0.40

# Challenge set
python3 test_seg.py --root ../dataset/HM \
                    --load ./model_seg.pth \
                    --subject '' \
                    --valid \
                    --model deeplab \
                    --conf_threshold 0.85 \
                    --val_threshold 0.40

python3 test_seg.py --root ../dataset/KL \
                    --load ./model_seg.pth \
                    --subject '' \
                    --valid \
                    --model deeplab \
                    --conf_threshold 0.85 \
                    --val_threshold 0.40
