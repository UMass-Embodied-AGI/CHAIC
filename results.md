# 🏆 Results

The table below shows the quantitative results of the CHAIC benchmark. We report the average Transport Rate (TR), Efficiency Improvement (EI), Goal Inference Accuracy (IA), Completion Ratio of Helper (CR), and Standard Error of Transport Rate (STD_TR) here. w/o means the main agent does the task solely without a helper. The Emergency Rate (ER) metric is also reported for the shopping task. 

<table>
    <tr>
        <td colspan="1"> <b>TR(EI)<span>&#8593;</span></b> </td>
        <td colspan="6" align="center">Indoor</td>
        <td colspan="2" align="center">Outdoor</td>
        <td colspan="1" align="center"></td>
    </tr>
    <tr>
        <td>Helper Agent</td>
        <td>Normal</td>
        <td>High Target</td>
        <td>High Container</td>
        <td>High Goalplace</td>
        <td>Lowthing</td>
        <td>Wheelchair</td>
        <td>Shopping</td>
        <td>Furniture</td>
        <td>Average</td>
    </tr>
    <tr>
        <td>w/o</td>
        <td>0.53</td>
        <td>0.30</td>
        <td>0.37</td>
        <td>0.28</td>
        <td>0.51</td>
        <td>0.07</td>
        <td>0.37</td>
        <td>0.17</td>
        <td>0.33</td>
    </tr>
    <tr>
        <td>Random</td>
        <td>0.52(-0.02)</td>
        <td>0.27(-0.05)</td>
        <td>0.36(0.00)</td>
        <td>0.33(0.10)</td>
        <td>0.50(-0.01)</td>
        <td>0.21(0.56)</td>
        <td>0.39(0.05)</td>
        <td>0.48(0.68)</td>
        <td>0.38(0.16)</td>
    </tr>
    <tr>
        <td>RHP</td>
        <td>0.64(0.15)</td>
        <td>0.35(0.11)</td>
        <td>0.45(0.19)</td>
        <td>0.35(0.18)</td>
        <td>0.66(0.23)</td>
        <td><b>0.44</b>(0.77)</td>
        <td>0.49(0.22)</td>
        <td>0.65(0.72)</td>
        <td>0.50(0.32)</td>
    </tr>
    <tr>
        <td>RL</td>
        <td>0.45(-0.19)</td>
        <td>0.26(-0.16)</td>
        <td>0.28(-0.25)</td>
        <td>0.25(-0.22)</td>
        <td>0.43(-0.16)</td>
        <td>0.11(0.07)</td>
        <td>0.32(-0.13)</td>
        <td>0.67(0.74)</td>
        <td>0.35(-0.04)</td>
    </tr>
    <tr>
        <td>SmartHelp</td>
        <td>0.46(-0.12)</td>
        <td>0.24(-0.17)</td>
        <td>0.26(-0.28)</td>
        <td>0.31(0.01)</td>
        <td>0.49(-0.04)</td>
        <td>0.13(0.11)</td>
        <td>0.32(-0.13)</td>
        <td>0.57(0.70)</td>
        <td>0.35(0.01)</td>
    </tr>
    <tr>
        <td>VLM</td>
        <td>0.63(0.14)</td>
        <td>0.33(0.06)</td>
        <td>0.43(0.12)</td>
        <td>0.26(-0.20)</td>
        <td>0.69(0.26)</td>
        <td>0.40(0.86)</td>
        <td>0.50(0.25)</td>
        <td><b>0.70(0.78)</b></td>
        <td>0.49(0.28)</td>
    </tr>
    <tr>
        <td>LLM+BM</td>
        <td><b>0.65(0.17)</b></td>
        <td><b>0.38(0.19)</b></td>
        <td><b>0.49(0.24)</b></td>
        <td><b>0.36(0.23)</b></td>
        <td><b>0.70(0.27)</b></td>
        <td>0.42(<b>0.89</b>)</td>
        <td><b>0.58(0.33)</b></td>
        <td>0.69(0.77)</td>
        <td><b>0.53(0.39)</b></td>
    </tr>
    <tr>
        <td>Oracle</td>
        <td>0.77(0.31)</td>
        <td>0.49(0.37)</td>
        <td>0.69(0.47)</td>
        <td>0.61(0.56)</td>
        <td>0.82(0.38)</td>
        <td>0.60(0.87)</td>
        <td>0.61(0.39)</td>
        <td>0.76(0.80)</td>
        <td>0.67(0.52)</td>
    </tr>
</table>
<table>
    <tr>
        <td colspan="1"> <b>IA<span>&#8593;</span></b> </td>
        <td colspan="6" align="center">Indoor</td>
        <td colspan="1" align="center">Outdoor</td>
        <td colspan="1" align="center"></td>
    </tr>
    <tr>
        <td>Helper Agent</td>
        <td>Normal</td>
        <td>High Target</td>
        <td>High Container</td>
        <td>High Goalplace</td>
        <td>Lowthing</td>
        <td>Wheelchair</td>
        <td>Shopping</td>
        <td>Average</td>
    </tr>
    <tr>
        <td>Random</td>
        <td>0.24</td>
        <td>0.29</td>
        <td>0.25</td>
        <td>0.14</td>
        <td>0.31</td>
        <td>0.24</td>
        <td>0.34</td>
        <td>0.26</td>
    </tr>
    <tr>
        <td>RHP</td>
        <td>0.15</td>
        <td>0.29</td>
        <td>0.21</td>
        <td>0.21</td>
        <td>0.28</td>
        <td>0.17</td>
        <td>0.44</td>
        <td>0.25</td>
    </tr>
    <tr>
        <td>VLM</td>
        <td>0.24</td>
        <td><b>0.32</b></td>
        <td><b>0.40</b></td>
        <td>0.33</td>
        <td><b>0.46</b></td>
        <td>0.35</td>
        <td>0.72</td>
        <td><b>0.40</b></td>
    </tr>
    <tr>
        <td>LLM+BM</td>
        <td><b>0.25</b></td>
        <td>0.29</td>
        <td>0.30</td>
        <td><b>0.35</b></td>
        <td>0.43</td>
        <td><b>0.47</b></td>
        <td><b>0.74</b></td>
        <td><b>0.40</b></td>
    </tr>
    <tr>
        <td>Oracle</td>
        <td>0.88</td>
        <td>0.91</td>
        <td>0.91</td>
        <td>0.90</td>
        <td>0.91</td>
        <td>0.82</td>
        <td>0.87</td>
        <td>0.89</td>
    </tr>
</table>
<table>
    <tr>
        <td colspan="1"> <b>CR<span>&#8593;</span></b> </td>
        <td colspan="6" align="center">Indoor</td>
        <td colspan="2" align="center">Outdoor</td>
        <td colspan="1" align="center"></td>
    </tr>
    <tr>
        <td>Helper Agent</td>
        <td>Normal</td>
        <td>High Target</td>
        <td>High Container</td>
        <td>High Goalplace</td>
        <td>Lowthing</td>
        <td>Wheelchair</td>
        <td>Shopping</td>
        <td>Furniture</td>
        <td>Average</td>
    </tr>
    <tr>
        <td>Random</td>
        <td>0.09</td>
        <td>0.10</td>
        <td>0.12</td>
        <td>0.06</td>
        <td>0.09</td>
        <td>0.09</td>
        <td>0.07</td>
        <td>0.73</td>
        <td>0.17</td>
    </tr>
    <tr>
        <td>RHP</td>
        <td>0.15</td>
        <td><b>0.43</b></td>
        <td>0.29</td>
        <td><b>0.39</b></td>
        <td>0.36</td>
        <td>0.19</td>
        <td>0.34</td>
        <td>0.74</td>
        <td>0.36</td>
    </tr>
    <tr>
        <td>VLM</td>
        <td>0.13</td>
        <td>0.08</td>
        <td><b>0.34</b></td>
        <td>0.18</td>
        <td><b>0.39</b></td>
        <td>0.17</td>
        <td>0.34</td>
        <td><b>0.82</b></td>
        <td>0.31</td>
    </tr>
    <tr>
        <td>LLM+BM</td>
        <td><b>0.22</b></td>
        <td>0.30</td>
        <td>0.30</td>
        <td>0.35</td>
        <td>0.38</td>
        <td><b>0.45</b></td>
        <td><b>0.46</b></td>
        <td>0.78</td>
        <td><b>0.41</b></td>
    </tr>
    <tr>
        <td>Oracle</td>
        <td>0.51</td>
        <td>0.64</td>
        <td>0.66</td>
        <td>0.73</td>
        <td>0.59</td>
        <td>0.38</td>
        <td>0.45</td>
        <td>0.77</td>
        <td>0.59</td>
    </tr>
</table>
<table>
    <tr>
        <td colspan="1"> <b>STD</b></td>
        <td colspan="6" align="center">Indoor</td>
        <td colspan="2" align="center">Outdoor</td>
        <td colspan="1" align="center"></td>
    </tr>
    <tr>
        <td>Helper Agent</td>
        <td>Normal</td>
        <td>High Target</td>
        <td>High Container</td>
        <td>High Goalplace</td>
        <td>Lowthing</td>
        <td>Wheelchair</td>
        <td>Shopping</td>
        <td>Furniture</td>
        <td>Average</td>
    </tr>
    <tr>
        <td>w/o</td>
        <td>0.03</td>
        <td>0.02</td>
        <td>0.03</td>
        <td>0.05</td>
        <td>0.03</td>
        <td>0.04</td>
        <td>0.02</td>
        <td>0.04</td>
        <td>0.03</td>
    </tr>
    <tr>
        <td>Random</td>
        <td>0.04</td>
        <td>0.03</td>
        <td>0.03</td>
        <td>0.04</td>
        <td>0.04</td>
        <td>0.04</td>
        <td>0.02</td>
        <td>0.05</td>
        <td>0.04</td>
    </tr>
    <tr>
        <td>RHP</td>
        <td>0.02</td>
        <td>0.04</td>
        <td>0.03</td>
        <td>0.05</td>
        <td>0.03</td>
        <td>0.04</td>
        <td>0.02</td>
        <td>0.04</td>
        <td>0.03</td>
    </tr>
    <tr>
        <td>VLM</td>
        <td>0.03</td>
        <td>0.02</td>
        <td>0.04</td>
        <td>0.05</td>
        <td>0.02</td>
        <td>0.03</td>
        <td>0.03</td>
        <td>0.05</td>
        <td>0.03</td>
    </tr>
    <tr>
        <td>LLM+BM</td>
        <td>0.03</td>
        <td>0.03</td>
        <td>0.03</td>
        <td>0.04</td>
        <td>0.03</td>
        <td>0.05</td>
        <td>0.03</td>
        <td>0.05</td>
        <td>0.04</td>
    </tr>
    <tr>
        <td>Oracle</td>
        <td>0.03</td>
        <td>0.04</td>
        <td>0.03</td>
        <td>0.04</td>
        <td>0.03</td>
        <td>0.03</td>
        <td>0.03</td>
        <td>0.04</td>
        <td>0.03</td>
    </tr>
</table>
<table>
    <tr>
        <td colspan="1"> <b>ER<span>&#8595;</span></b> </td>
        <td colspan="1" align="center">Outdoor</td>
    </tr>
    <tr>
        <td>Helper Agent</td>
        <td>Shopping</td>
    </tr>
    <tr>
        <td>Random</td>
        <td>0.32</td>
    </tr>
    <tr>
        <td>RHP</td>
        <td><b>0.30</b></td>
    </tr>
    <tr>
        <td>VLM</td>
        <td>0.39</td>
    </tr>
    <tr>
        <td>LLM+BM</td>
        <td>0.38</td>
    </tr>
    <tr>
        <td>Oracle</td>
        <td>0.17</td>
    </tr>
</table>