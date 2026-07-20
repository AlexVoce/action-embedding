

PCOMPBIOL-D-26-00710  
Why motor learning involves multiple systems: an algorithmic perspective  
PLOS Computational Biology  
  
Dear Dr. Clopath,  
  
Thank you for submitting your manuscript to PLOS Computational Biology. After careful consideration, we feel that it has merit but does not fully meet PLOS Computational Biology's publication criteria as it currently stands. Therefore, we invite you to submit a revised version of the manuscript that addresses the points raised during the review process.  
  
Please submit your revised manuscript by Sep 05 2026 11:59PM. If you will need more time than this to complete your revisions, please reply to this message or contact the journal office at [ploscompbiol@plos.org](mailto:ploscompbiol@plos.org). When you're ready to submit your revision, log on to [https://www.editorialmanager.com/pcompbiol/](https://www.editorialmanager.com/pcompbiol/) and select the 'Submissions Needing Revision' folder to locate your manuscript file.  
  
Please include the following items when submitting your revised manuscript:  
  
* A letter that responds to each point raised by the editor and reviewer(s). You should upload this letter as a separate file labeled 'Response to Reviewers'. This file does not need to include responses to formatting updates and technical items listed in the 'Journal Requirements' section below.  
* A marked-up copy of your manuscript that highlights changes made to the original version. You should upload this as a separate file labeled 'Revised Manuscript with Track Changes'.  
* An unmarked version of your revised paper without tracked changes. You should upload this as a separate file labeled 'Manuscript'.  
  
If you would like to make changes to your financial disclosure, competing interests statement, or data availability statement, please make these updates within the submission form at the time of resubmission. Guidelines for resubmitting your figure files are available below the reviewer comments at the end of this letter.

As the corresponding author, your ORCID iD is verified in the submission system and will appear in the published article. PLOS supports the use of ORCID, and we encourage all coauthors to register for an ORCID iD and use it as well. Please encourage your coauthors to verify their ORCID iD within the submission system before final acceptance, as unverified ORCID iDs will not appear in the published article. Only the individual author can complete the verification step; PLOS staff cannot verify ORCID iDs on behalf of authors.

We look forward to receiving your revised manuscript.  
  
Kind regards,  
  
Jorge F. Mejias, Ph.D.  
Academic Editor  
PLOS Computational Biology  
  
Tianming Yang  
Section Editor  
PLOS Computational Biology  
  
  
**Additional Editor Comments (if provided):**  
  
  
  
  
**Journal Requirements:**  
  
If the reviewer comments include a recommendation to cite specific previously published works, please review and evaluate these publications to determine whether they are relevant and should be cited. There is no requirement to cite these works unless the editor has indicated otherwise.

1) Please ensure that the CRediT author contributions listed for every co-author are completed accurately and in full.  
At this stage, the following Authors/Authors require contributions: Francesca Greenstreet, Jesse Geerts, Juan Gallego, and Claudia Clopath. Please ensure that the full contributions of each author are acknowledged in the "Add/Edit/Remove Authors" section of our submission form.  
The list of CRediT author contributions may be found here: [https://journals.plos.org/ploscompbiol/s/authorship#loc-author-contributions](https://journals.plos.org/ploscompbiol/s/authorship#loc-author-contributions)

2) We ask that a manuscript source file is provided at Revision. Please upload your manuscript file as a .doc, .docx, .rtf or .tex. If you are providing a .tex file, please upload it under the item type u2018LaTeX Source Fileu2019 and leave your .pdf version as the item type u2018Manuscriptu2019.

3) Please provide an Author Summary. This should appear in your manuscript between the Abstract (if applicable) and the Introduction, and should be 150-200 words long. The aim should be to make your findings accessible to a wide audience that includes both scientists and non-scientists. Sample summaries can be found on our website under Submission Guidelines:  
[https://journals.plos.org/ploscompbiol/s/submission-guidelines#loc-parts-of-a-submission](https://journals.plos.org/ploscompbiol/s/submission-guidelines#loc-parts-of-a-submission)

4) Please upload all main figures as separate Figure files in .tif or .eps format. For more information about how to convert and format your figure files please see our guidelines:   
[https://journals.plos.org/ploscompbiol/s/figures](https://journals.plos.org/ploscompbiol/s/figures)

5) Please upload a copy of Figures FIG 1 A-K, 2A-D, 3A-I, and 4A-G which you refer to in your text on pages 3, 8, 11, and 14. Or, if the figure is no longer to be included as part of the submission please remove all reference to it within the text.

6) We notice that your supplementary Figures are included in the manuscript file. Please remove them and upload them with the file type 'Supporting Information'. Please ensure that each Supporting Information file has a legend listed in the manuscript after the references list.

7) Please amend your detailed Financial Disclosure statement. This is published with the article. It must therefore be completed in full sentences and contain the exact wording you wish to be published.  
i) State the initials, alongside each funding source, of each author to receive each grant. For example: "This work was supported by the National Institutes of Health (####### to AM; ###### to CJ) and the National Science Foundation (###### to AM)."  
ii) State what role the funders took in the study. If the funders had no role in your study, please state: "The funders had no role in study design, data collection and analysis, decision to publish, or preparation of the manuscript."  
iii) If any authors received a salary from any of your funders, please state which authors and which funders..  
If you did not receive any funding for this study, please simply state: u201cThe authors received no specific funding for this work.u201d

8)Please send a completed 'Competing Interests' statement, including any COIs declared by your co-authors. If you have no competing interests to declare, please state "The authors have declared that no competing interests exist". Otherwise please declare all competing interests beginning with the statement "I have read the journal's policy and the authors of this manuscript have the following competing interests.

.  
**Reviewers' comments:**  
  
Reviewer's Responses to Questions

**Comments to the Authors:  
Please note here if the review is uploaded as an attachment.**

Reviewer #1: The authors propose a novel computational framework for motor learning in which the cortico-cerebellar network learns low-dimensional action embeddings via supervised learning, which the basal ganglia then exploit for reinforcement learning. This is a timely and interesting idea that connects a machine learning literature on action representations to well-studied neural and behavioural phenomena in motor control. The model is shown to recapitulate specification-like activity in the basal ganglia, generalisation of visuomotor adaptation, and limits on dual adaptation.  
  
However, the paper's core claims rest on a set of modelling choices that are not fully justified, and in several cases the connection to the experimental data being modelled is looser than presented. The Fourier state representation pre-encodes the spatial structure the supervised network is credited with discovering; the tasks are exclusively single-step; and the efficacy of the embedding approach over simpler baselines is asserted but not demonstrated. The fits to Park 2025 and Woolley 2007 also involve parameter choices and phase alignments that are not well motivated. Taken together, these issues make it difficult to evaluate the specific contribution of the proposed architecture. I recommend major revision.  
  
  
# Major issues  
  
## Conceptual gaps  
  
### Slow RL in BG vs Fast SL in cortex/cerebellum  
  
> “As motor adaptation is a fast cerebellar-dependent process based on sensorimotor errors [29] that is not disrupted by basal ganglia dysfunction according to studies in Parkinsonian patients [70–72], we modelled this fast adaptation process as happening through supervised learning in the corticocerebellar network, rather than through reinforcement learning in the basal ganglia, which is slow [41].”  
  
The paper calls RL "slow" and SL "fast", but this gets a bit confusing because some of the citations (Wang 2018 especially) propose meta-learning RL strategies that support rapid learning via neural activity (where strategies are learned via slower weight/synaptic change). I don’t think motor learning is supported by the fast meta-learned algorithms described in Wang 2018, but this seems worth clarifying.  
  
While the classical view seems to be that the cerebellum supports rapid adaptation, and is well-suited to supervised learning (like in the cited Doya 2000), this framing seems less clear for cortex which is typically thought to learn slowly via Hebbian updates. The discussion acknowledges some of these conceptual issues, but it should be foregrounded in the introduction. Otherwise, the paper can be read as advocating that rapid motor adaptation happens via changes in both cortex and cerebellum.  
  
  
## Ensuring robustness of the modeling results  
  
### State representations are closely related to desired action representations.  
  
> “For simplicity we model the state representation of each reach location using a third order Fourier Basis. This uniquely identifies each possible task state, whilst ensuring that similar reach endpoints have more similar state representations, a property that is critical to learn meaningful embeddings.”  
This state representation seems to straightforwardly allow reconstruction of x/y coordinates, or dx/dy actions. This is worth remarking on -- despite the lack of structure in the action space, the state representation can still induce / force structure. For example, using the paper's code I found that the top dimensions of a PCA of state differences $phi(s_{t+1})-phi(s_t)$ replicate the action structure shown in the paper.  
  
I’m not sure what to recommend here. My concern is that the state representation is only a slightly obfuscated x/y coordinate, and this makes the desired action embedding easy to learn. I can imagine that one-hot representations of space (requiring something like a grid world) are one alternative, and demonstrating that meaningful action representations can still be learned here would be valuable. However, discretizing space (for the purpose of having a tractable number of states) invites another issue – the action space becomes quite small. Even if only in the supplement, this would be rhetorically helpful in justifying the overall method and minimizing undue influence of the Fourier state representation.  
  
### Single-step RL  
  
> “We model the centre-out reaching task as a single step reinforcement learning environment where, on each trial, the agent starts at a location in the middle of the environment. There are 24 available actions (length 1) at equally spaced angles. Each action deterministically causes a state transition to one of 24 reach endpoints in the environment (Figure 1C).”  
The tasks in the paper are largely (or entirely?) single-step RL tasks. I’m concerned that this is overly simplistic, making it hard to judge the specific contributions of learning via supervision vs. reward. In particular, given the Gaussian policy, single-step, and single-target nature of the model, it essentially fits the mean of the Gaussian to rewarded embeddings. This lacks the sequential decisions that typically characterize RL. I encourage the authors to acknowledge this more explicitly early on and, even if only for the supplement, add an experiment with sequential action, by producing a smaller action offset that requires multiple steps to reach the goal.  
  
### Baseline comparisons  
  
> “This approach drastically speeds up learning by improving how the space of potential actions is explored, and leads to better alignment between action space and the task [43, 46].”  
This claim about the efficacy of the method is not substantiated in the text. It would be worth showing that the method accelerates learning beyond a simpler policy gradient baseline method.  
  
> “the basal ganglia could learn more efficiently in a lower-dimensional space structured by the supervised learning system.”  
An open question to me is to what extent these findings could be replicated by a fully RL-based system that has a simple low-rank projection / bottleneck. I suspect this would recover a similar action structure (given the Fourier state representation already encodes spatial proximity; see comment on state representation below), but it is unclear whether it would show the same adaptation and interference profiles. Nonetheless, it seems worthwhile to include as a baseline.  
  
  
## Further details about modeled experiments  
  
### Park 2025  
  
The experiments in Figure 2C rely on large offsets, which Park seems to intentionally avoid ("Thus, a behavioral paradigm in which an animal is reliably selecting between actions with small parameter variation is uniquely useful for disambiguating the expected neural correlates between continuous (specification) or discrete (selection) action representations."). Given that the Park figure lacks the large offset the authors have tried, the authors should either 1) directly simulate Park’s conditions, or 2) more explicitly discuss why the large offset condition is included despite Park’s theoretical reasons for excluding it.  
  
### Woolley 2007  
  
The figures and experiment should more closely reflect (or describe) the original Woolley 2007 procedures. I detail some quick thoughts based on the original paper, and apologize for any misunderstanding on my part.  
  
- The targets were not "indistinguishable". Per Woolley: "For Group 1, the direction of rotation of the feedback was specified only by the red and blue display background colour cues"  
- Figures 4A,B should note that "A black cursor path represents an initial trial in a block, with a grey cursor path representing a final trial."  
- Woolley 2007 had a multi-stage procedure: PRE was a familiarization phase, TRAINING had rotated feedback, and POST removed rotation from feedback. I understand that it's verbose to detail this, but the presented results aren't clear about the phase Figures 4A,B are from, and how the displayed lines should correspond to the model. The presented results also seem to come from the POST phase, but the TRAINING phase seems to be most analogous to the simulation's focus on rotated feedback. The figure caption should clarify which phase each panel represents, and the correspondence to the model made explicit.  
  
  
  
# Minor issues  
  
> “in robotics there are multiple joint configurations that can achieve the same end-effector position or movement outcome”  
I think this is classically referred to in motor control, as well as robotics, as "motor equivalence".  
  
Figure 2: Reference 63 is to the wrong paper.  
  
Figure 3I: Inverted y-axis is confusing compared to 3E which also has an angular error -- label it "negative angular error" and change sign? Better yet -- may be worth plotting a similar quantity as Krakauer 2000, which seems to plot a transformed angular error per their methods. In either case, it is worth clarifying what Krakauer’s reported quantity is.  
  
> “In stark contrast to this traditional view, a recent behavioural study [100], further supported by our modelling work [35], suggests that motor adaptation is best described based on directly updating the control policy (i.e., the inverse model), without the need to invert an updated forward model.”  
Given that the presented work departs from this prevailing view, it merits further discussion.  
  
> “In our model, the supervised learning system predicts the action that would cause a particular state transition, bearing strong similarity to a classic inverse model, and thus most strongly aligns to this direct control policy update view.”  
It is worth being explicit that a typical inverse model is goal/reward-conditioned, while the present embedding model focuses more specifically on inverting the transition model.  
  
The std update equation (if it mirrors the radius update equation) should be $std <- **std_max** - $  
  
“Embedding learning rate αe” should be reported for the "Embedding Learning" column.

Reviewer #2: Summary  
The manuscript asks why motor learning relies on multiple interacting systems rather than a single learning mechanism. The authors propose that cortex and cerebellum learn a structured representation of the action space through supervised learning, while the basal ganglia learn action values and policies within that representation using reinforcement learning. They show that this framework can account for several experimental observations, including specification-like striatal activity, the local generalisation of visuomotor adaptation, and limits on simultaneous dual adaptation.  
The manuscript's main strength is the attempt to explain these diverse phenomena within a single computational framework. In particular, the proposal that adaptation generalisation and dual-adaptation interference arise from the same property of the action-embedding mapping provides a simple and, to my knowledge, novel synthesis of previously separate observations.  
My concerns are not with the modelling itself but with the relationship between the biological claims and the evidence presented. The framework relies on a mapping of supervised learning onto the cortico-cerebellar network and reinforcement learning onto the basal ganglia, but the assumptions underlying this mapping remain underspecified. The manuscript does not commit to a cortex/cerebellum division of labour, does not engage known cerebellum–basal ganglia pathways, and does not directly test the high-dimensional scaling argument that motivates the framework. In addition, several quantitative matches may depend on tuned hyperparameters rather than constituting independent predictions. These issues do not undermine the central idea, but clarifying the scope of the biological interpretation would substantially strengthen the manuscript.  
  
Major points  
1. The title, abstract and introduction foreground the interaction of motor cortex, cerebellum and basal ganglia, but the model itself is formulated as an interaction between supervised and reinforcement learning. While the reinforcement-learning component is explicitly associated with the basal ganglia, the cortico-cerebellar component is represented by a generic supervised-learning system rather than by distinct cortical and cerebellar elements. As a result, it is difficult to determine which conclusions follow directly from the computational framework and which depend on the assumed mapping between the supervised system and the cortico-cerebellar network. Distinguishing more clearly between these two levels of explanation would strengthen the biological interpretation.  
2. I understand from Figure 1 that in the embedding-learning narrative the cortico-cerebellar system learns the embedding (the encoder g, which holds the low-dimensional structure). In the adaptation results (Figures 3–4) g is frozen and only the decoder f is plastic, and this f-plasticity is explicitly motivated as fast cerebellar adaptation. What remains unclear is how these components map onto the proposed roles of cortex and cerebellum. In the adaptation results, the cerebellum appears to correspond to the decoder f, whereas in the Discussion the cerebellum is proposed to learn the action embedding itself, which would instead associate it with the encoder g. Therefore, the cerebellum is associated with the embedding in one place and with the adaptation mechanism in another, leaving the role of cortex underspecified.  
3. The manuscript motivates the framework through interactions between cortex, cerebellum and basal ganglia, yet the proposed interaction is effectively mediated through the cortico-cerebellar representation. Anatomical studies have identified disynaptic pathways linking cerebellum and basal ganglia, including projections from the dentate nucleus to striatum and from the subthalamic nucleus to cerebellar cortex. The first pathway is particularly relevant because the model places reinforcement learning and the specification-like activity of Figure 2 in the striatum, which receives a cortex-independent cerebellar projection. This raises the possibility that cerebellar output could influence basal-ganglia learning through routes that are not explicitly considered in the current framework.  
4. The paper opens on vast action spaces, hundreds of joints and muscles, and robotic control, but every result is obtained in a single-DOF, 24-action, single-step task whose embedding is a 2D circle. The scaling advantage, which motivates the proposed architecture, is supported primarily by the parameter-count analysis of Figure A.1 and by prior work (Chandak et al.), rather than by a direct demonstration of improved learning speed within the present model. The Discussion appropriately acknowledges this limitation and frames the relevance to higher-dimensional problems as a prediction rather than a demonstrated result. The same distinction is less apparent in the abstract and introduction.  
5. The model is evaluated in a single-step task, in which each action produces a single terminal transition and an immediate reward. It seems to me that this makes the reinforcement-learning problem closer to a contextual bandit than to a sequential control problem. It does not invalidate the central idea, but it limits the extent to which conclusions about temporal-difference learning and basal-ganglia function can be drawn, since the temporal-difference component plays only a limited role when every episode terminates after one step. I therefore encourage the authors either to acknowledge this limitation more explicitly or to discuss how the framework might extend to multi-step settings.  
6. One of the manuscript's most interesting findings is the emergence of specification-like striatal activity. However, because the learned embedding is explicitly encouraged to place actions with similar consequences near one another, some degree of similarity between nearby actions is expected from the learned representational structure itself. It is therefore difficult to determine which aspects of the result arise from reinforcement learning within the embedding space and which follow more directly from the properties of the embedding. In my view, the stronger and more novel claim is not that neighbouring actions are represented similarly, but that this structure is learned through interaction with the environment rather than imposed a priori.  
7. I found the selection baseline in Figure 2 difficult to interpret. As far as I can tell, the "selection encoding model" is replotted from an in-context-learning paper rather than from a model of striatal selection. If this is correct, the comparison should either be justified more explicitly or replaced with a baseline that more directly reflects selection-based accounts of basal-ganglia function.  
8. The manuscript reproduces several behavioural phenomena with impressive qualitative fidelity. However, the spatial scales of adaptation generalisation (Figure 3I) and dual-adaptation interference (Figure 4G) depend on parameters that were tuned by hand. It is difficult to determine whether the observed scales constitute independent predictions of the framework or reflect particular modelling choices.  
  
Minor points  
1. The reach-to-pull task of Park et al. is reduced to a centre-out reaching task, which is a reasonable simplification, but the manuscript should state more clearly in the main text (and not only in the Methods) that the comparison is qualitative. In addition, the small-separation condition is described as 30° "as in Park et al.", whereas the targets in Park et al. were separated by 10° and the widening to 30° was introduced here for visualisation purposes.  
2. M2/ACA (Figure 2) is not defined.  
3. The discussion of Parkinson's disease and adaptation could be more nuanced. It seems to me that the cited studies (refs 70–72) support intact initial adaptation, but also report impairments in savings, consolidation or retention.  
4. The proposed supervised-learning module is closer to an inverse mapping than to a classical forward model, and the manuscript already discusses this distinction in relation to the literature. However, I do not think the implications of this comparison are fully developed. In the classical forward/inverse-model framework, the inverse model directly generates motor commands. Here, by contrast, the supervised module is not used as a controller but as a mechanism for learning a structured action representation. As written, the manuscript establishes the resemblance to an inverse model, but not the significance of assigning it a different computational role.

---

 

**Have the authors made all data and (if applicable) computational code underlying the findings in their manuscript fully available?**  
The [PLOS Data policy](https://track.editorialmanager.com/CL0/https:%2F%2Fjournals.plos.org%2Fploscompbiol%2Fs%2Fmaterials-and-software-sharing/1/010f019f3725def1-ff39dd88-dbfa-4b1d-bd3f-5c24d2a3ff50-000000/g3I64wfZl25lcTzf930A3yepb-3EwEPy7goTdNSgIGI=258) requires authors to make all data and code underlying the findings described in their manuscript fully available without restriction, with rare exception (please refer to the Data Availability Statement in the manuscript PDF file). The data and code should be provided as part of the manuscript or its supporting information, or deposited to a public repository. For example, in addition to summary statistics, the data points behind means, medians and variance measures should be available. If there are restrictions on publicly sharing data or code —e.g. participant privacy or use of data from a third party—those must be specified.

Reviewer #1: Yes

Reviewer #2: Yes

---

 

PLOS authors have the option to publish the peer review history of their article ([what does this mean?](https://track.editorialmanager.com/CL0/https:%2F%2Fjournals.plos.org%2Fploscompbiol%2Fs%2Feditorial-and-peer-review-process%23loc-peer-review-history/1/010f019f3725def1-ff39dd88-dbfa-4b1d-bd3f-5c24d2a3ff50-000000/IdPF4Fcyzf9Kbfbn5SJpXVjRyw5QaDG0iCPdr1f-5zw=258)). If published, this will include your full peer review and any attached files.  
  
  
If you choose “no”, your identity will remain anonymous but your review may still be made public.  
  
  
**Do you want your identity to be public for this peer review?** For information about this choice, including consent withdrawal, please see our [Privacy Policy](https://track.editorialmanager.com/CL0/https:%2F%2Fwww.plos.org%2Fprivacy-policy/1/010f019f3725def1-ff39dd88-dbfa-4b1d-bd3f-5c24d2a3ff50-000000/nYpR6CxPEIEGj7hPIThmLaMQEPMDAAUqVscPGIFpf9k=258).

Reviewer #1: No

Reviewer #2: **Yes:** Elías Mateo Fernández Santoro

  
  
[NOTE: If reviewer comments were submitted as an attachment file, they will be attached to this email and accessible via the submission site. Please log into your account, locate the manuscript record, and check for the action link "View Attachments". If this link does not appear, there are no attachment files.]  
  
**Figure resubmission:**  
  
While revising your submission, we strongly recommend that you use PLOS’s NAAS tool ([https://ngplosjournals.pagemajik.ai/artanalysis](https://ngplosjournals.pagemajik.ai/artanalysis)) to test your figure files. NAAS can convert your figure files to the TIFF file type and meet basic requirements (such as print size, resolution), or provide you with a report on issues that do not meet our requirements and that NAAS cannot fix.

  
After uploading your figures to PLOS’s NAAS tool - [https://ngplosjournals.pagemajik.ai/artanalysis](https://ngplosjournals.pagemajik.ai/artanalysis), NAAS will process the files provided and display the results in the "Uploaded Files" section of the page as the processing is complete. If the uploaded figures meet our requirements (or NAAS is able to fix the files to meet our requirements), the figure will be marked as "fixed" above. If NAAS is unable to fix the files, a red "failed" label will appear above. When NAAS has confirmed that the figure files meet our requirements, please download the file via the download option, and include these NAAS processed figure files when submitting your revised manuscript.

  
**Reproducibility:**  
  
To enhance the reproducibility of your results, we recommend that authors of applicable studies deposit laboratory protocols in [protocols.io](http://protocols.io/), where a protocol can be assigned its own identifier (DOI) such that it can be cited independently in the future. Additionally, PLOS ONE offers an option to publish peer-reviewed clinical study protocols. Read more information on sharing protocols at [https://plos.org/protocols?utm_medium=editorial-email&utm_source=authorletters&utm_campaign=protocols](https://plos.org/protocols?utm_medium=editorial-email&utm_source=authorletters&utm_campaign=protocols)

  
  

---

In compliance with data protection regulations, you may request that we remove your personal registration details at any time.  [(Remove my information/details)](https://track.editorialmanager.com/CL0/https:%2F%2Fwww.editorialmanager.com%2Fpcompbiol%2Flogin.asp%3Fa=r/1/010f019f3725def1-ff39dd88-dbfa-4b1d-bd3f-5c24d2a3ff50-000000/LJT3g1eeN6ByYT82rS6T7fHoOlWO8P4BgJHdMXgEzzA=258). Please contact the publication office if you have any questions.