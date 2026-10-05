{{- define "oxllm.config" -}}
{{- $config := deepCopy .Values.config -}}
{{- if .Values.persistence.enabled -}}
{{- $general := get $config "general_settings" | default dict -}}
{{- $_ := set $general "database_url" "os.environ/OXLLM_DATABASE_URL" -}}
{{- $_ := set $config "general_settings" $general -}}
{{- end -}}
{{- toYaml $config -}}
{{- end -}}

{{- define "oxllm.collectEnvRefs" -}}
{{- if or (kindIs "map" .value) (kindIs "slice" .value) -}}
{{- range $value := .value -}}
{{- include "oxllm.collectEnvRefs" (dict "value" $value "refs" $.refs) -}}
{{- end -}}
{{- else if kindIs "string" .value -}}
{{- if hasPrefix "os.environ/" .value -}}
{{- $_ := set .refs (trimPrefix "os.environ/" .value) true -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "oxllm.drainSeconds" -}}
{{- $general := .Values.config.general_settings | default dict -}}
{{- $settings := .Values.config.oxllm_settings | default dict -}}
{{- $shutdown := 30 -}}
{{- $request := 600 -}}
{{- if hasKey $general "shutdown_timeout_secs" -}}
{{- $shutdown = int $general.shutdown_timeout_secs -}}
{{- end -}}
{{- if hasKey $settings "request_timeout" -}}
{{- $request = int $settings.request_timeout -}}
{{- end -}}
{{- max $shutdown $request -}}
{{- end -}}

{{- define "oxllm.graceSeconds" -}}
{{- $drain := include "oxllm.drainSeconds" . | int -}}
{{- if kindIs "invalid" .Values.terminationGracePeriodSeconds -}}
{{- add $drain 30 -}}
{{- else -}}
{{- if le (int .Values.terminationGracePeriodSeconds) $drain -}}
{{- fail (printf "terminationGracePeriodSeconds must exceed the server drain budget (%ds); leave time for spend/telemetry flushing" $drain) -}}
{{- end -}}
{{- .Values.terminationGracePeriodSeconds -}}
{{- end -}}
{{- end -}}
