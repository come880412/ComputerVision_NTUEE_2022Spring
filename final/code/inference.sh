# public set
python3 test_seg.py --root ../dataset/ \
                    --load_seg ./model_seg.pth \
                    --load_valid ./model_valid.pth \
                    --subject S5 \
                    --valid \
                    --model deeplab \
                    --conf_threshold 0.85 \
                    --val_threshold 0.40

# Challenge set
python3 test_seg.py --root ../dataset/HM \
                    --load_seg ./model_seg_chanllenge.pth \
                    --subject '' \
                    --model deeplab \
                    --conf_threshold 0.85 \

python3 test_seg.py --root ../dataset/KL \
                    --load_seg ./model_seg_chanllenge.pth \
                    --subject '' \
                    --model deeplab \
                    --conf_threshold 0.85 \
