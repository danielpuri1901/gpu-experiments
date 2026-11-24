# k8s-model-service

This exercise asks one question: how does Kubernetes keep a model microservice available while it crashes, updates and grows, and what does a service mesh add?
It covers exam objective enterprise.1 of NVIDIA NCA-AMAD: service deployment and orchestration, containers, Kubernetes, model and microservice deployment, infrastructure as code and service meshes.
A small Python server stands in for a NIM, because a real NIM needs a GPU node and GitHub-hosted runners have none.
The server has OpenAI-style routes, burns CPU per request, and reports its version and pod name.
Everything runs only in GitHub Actions, on a kind cluster with one control plane and two workers.
No image is built: the code comes from a ConfigMap and runs in a pinned `python:slim` image.

## What each test shows

| Kubernetes concept | What the test shows | Exam rule |
| --- | --- | --- |
| Readiness probe | Scale from 3 to 6 pods while each pod loads its model for 15 s. Count failed requests with the probe, and without it as a control. | Readiness decides who gets traffic. Liveness decides who gets restarted. |
| ReplicaSet and liveness probe | Delete a pod under load, then break another pod with `/break`. Measure the time to 3 ready pods and the time to the restart. | A controller keeps the actual state equal to the desired state. |
| Rolling update, `maxUnavailable: 0`, rollback | Update v1 to v2 under load. Then ship a v3 that never gets ready, and run `kubectl rollout undo`. | A bad release stalls and the old pods keep serving. Undo restores the last good revision. |
| HorizontalPodAutoscaler, metrics-server | CPU load raises replicas from 3 towards 6. Then measure the scale-in delay after the load stops. | The HPA needs resource requests and a metrics source. Scale-in waits for a stabilisation window. A real NIM scales on a GPU metric, not CPU. |
| Istio VirtualService and DestinationRule | Send 90/10, then 50/50, then 100% of traffic to v2 and measure the v2 share. | A mesh splits traffic by weight, not by replica count. |
| Outlier detection | Poison one pod so it returns 500. Compare failed requests without and with ejection. | Outlier detection is the mesh circuit breaker. It ejects a failing endpoint. |
| PeerAuthentication STRICT | A pod outside the mesh calls the service in plain text, before and after STRICT. A pod inside the mesh calls it too. | STRICT mTLS refuses plain text. PERMISSIVE accepts both. |
| Kustomize, PodDisruptionBudget | The same plain YAML is applied to two namespaces. The PDB is part of the manifests but no test uses it. | Describe the desired state in files, not in commands. |

## How to run

1. Open the Actions tab of the repository.
2. Choose the `k8s-model-service` workflow and select Run workflow.
3. A push to main that changes `k8s-model-service/**` or the workflow file starts it too.
4. Read the table in the job summary. Each check shows PASS or FAIL and the measured number.
5. Download the artifact `k8s-model-service-results` for `results.json`, the raw load-generator output and the logs.

The job fails if one check fails.
The artifact is uploaded in both cases.
On a failure the job also dumps events, pod descriptions and pod logs.

## Files

- `base/app.yaml`: Deployment, Service and PodDisruptionBudget.
- `hpa.yaml`: HorizontalPodAutoscaler. Only test 4 applies it.
- `base/server.py`: the model server. `base/loadgen.py`: the load generator and the result analysis.
- `mesh/`: the Istio VirtualService, DestinationRules and the STRICT policy.
- `kind.yaml`: the cluster. `tests.sh`: the six tests.

## Limits

- The server is not a NIM. It shows Kubernetes and mesh behaviour, not GPU behaviour.
- The whole cluster shares one runner with 4 vCPU. Latency numbers are noisy and one run is one sample.
- Timings come from polling once per second. Read them as plus or minus 1 s.
- The HPA uses CPU. A NIM would use a GPU metric such as KV cache usage, which needs Prometheus and an adapter.
- The mesh uses Istio sidecar mode with the minimal profile. Ambient mode is not tested.
- The PodDisruptionBudget has no test.
